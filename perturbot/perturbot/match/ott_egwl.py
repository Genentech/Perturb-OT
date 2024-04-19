# Modified OTT with labels
from typing import Dict, Tuple
from numbers import Number
import numpy as np
import jax.numpy as jnp
import time
import ott
from ott.geometry import pointcloud, geometry
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers import linear
from ott.solvers.linear import acceleration, sinkhorn


def create_block_diag_mat(labels_a, labels_b):
    block_diag_mat = np.zeros((len(labels_a), len(labels_b)))
    for l in np.unique(labels_a):
        block_diag_mat[
            np.ix_(np.where(labels_a == l)[0], np.where(labels_b == l)[0])
        ] = 1.0
    return jnp.array(block_diag_mat)


def get_coupling_egw_labels_ott(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]], eps: float = 5e-3
) -> Tuple[Dict[Number, np.array], Dict]:
    r"""Returns GW coupling between two datasets X, Y given the labels.

    The function solves the following optimization problem:

    .. math::

        EGWL = \min_{T\in C_{p,q}^\ell} \sum_{i,k \in \{i|l_{x_i}=t\}, j,l \in \{j|l_{y_j}=t\}} |(x_i-x_k)^2 - (y_j-y_l)^2|^{2}*T_{i,j}T_{k,l} - \epsilon H(T)\\
        C_{p,q}^\ell = \{T | T \in C{p,q}, T_{ij} > 0 \implies l_{x_i} = l_{y_j}\}

    Parameters
    ----------
    data : 
        (source dataset, target dataset) where source and target datasets 
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps: 
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict : 
        Optimal Transport coupling between the samples per label
    log : 
        Running log
    
    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_egw_labels_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_egw_labels_ott((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    Xs_tot = jnp.array(np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0))
    Xt_tot = jnp.array(np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0))
    source_labels = jnp.array(
        np.concatenate([np.repeat(l, X_dict[l].shape[0]) for l in X_dict.keys()])
    )
    target_labels = jnp.array(
        np.concatenate([np.repeat(l, Y_dict[l].shape[0]) for l in X_dict.keys()])
    )
    start = time.time()

    geom_xx = pointcloud.PointCloud(x=Xs_tot, y=Xs_tot, scale_cost="max_cost")
    geom_yy = pointcloud.PointCloud(x=Xt_tot, y=Xt_tot, scale_cost="max_cost")
    bdm = create_block_diag_mat(source_labels, target_labels)

    cost_time = time.time() - start
    start = time.time()
    print("running EGWL with ott")

    prob = quadratic_problem.QuadraticProblem(
        geom_xx,
        geom_yy,
        labels_a=source_labels,
        labels_b=target_labels,
        n_labels=len(np.unique(source_labels)),
        block_diag_mat=bdm,
    )

    solver = gromov_wasserstein.GromovWasserstein(
        epsilon=eps,
        store_inner_errors=True,
        max_iterations=2000,
        kwargs_sinkhorn={"max_iterations": 2000},
    )

    out = solver(prob)

    has_converged = bool(out.linear_convergence[out.n_iters - 1])
    log = {}
    log["n_iters_outer"] = out.n_iters
    log["converged_inner"] = has_converged
    log["converged_outer"] = out.converged
    log["GW cost"] = out.reg_gw_cost
    T = np.array(out.matrix)
    print(f"{out.n_iters} outer iterations were needed.")
    print(f"The last Sinkhorn iteration has converged: {has_converged}")
    print(f"The outer loop of Gromov Wasserstein has converged: {out.converged}")
    print(f"The final regularized GW cost is: {out.reg_gw_cost:.3f}")

    end = time.time()
    print("Done running LEGWOT with ott")
    log["time"] = end - start
    log["cost_time"] = cost_time
    T_dict = {}
    for l in np.unique(source_labels):
        T_dict[l] = T[np.array(source_labels) == l, :][:, np.array(target_labels) == l]
    return T_dict, log


def get_coupling_egw_ott(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]], eps: float = 5e-3
) -> Tuple[Dict[Number, np.array], Dict]:
    r"""Returns GW coupling between two datasets X, Y per label.

    The function solves the following optimization problem:

    .. math::

        GW^l = \min_{T^l} \sum_{i,j,k,l} |(x_i-x_k)^2 - (y_j-y_l)^2|^{2}*T^l_{i,j}T^l_{k,l} - \epsilon H(T^l)\\
    

    Parameters
    ----------
    data : 
        (source dataset, target dataset) where source and target datasets 
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps: 
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict : 
        Optimal Transport coupling between the samples per label
    log : 
        Running log
    
    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_egw_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_egw_ott((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    labels = X_dict.keys()
    Ts = {}
    log = {}
    for l in labels:
        log[l] = {}
        start = time.time()

        geom_xx = pointcloud.PointCloud(x=X_dict[l], y=X_dict[l], scale_cost="max_cost")
        geom_yy = pointcloud.PointCloud(x=Y_dict[l], y=Y_dict[l], scale_cost="max_cost")

        cost_time = time.time() - start
        start = time.time()

        prob = quadratic_problem.QuadraticProblem(
            geom_xx,
            geom_yy,
        )

        # Instantiate a jitt'ed Gromov-Wasserstein solver
        solver = gromov_wasserstein.GromovWasserstein(
            epsilon=eps, store_inner_errors=True, max_iterations=1000
        )

        out = solver(prob)

        end = time.time()
        Ts[l] = np.array(out.matrix)

        has_converged = bool(out.linear_convergence[out.n_iters - 1])
        log[l]["n_iters_outer"] = out.n_iters
        log[l]["converged_inner"] = has_converged
        log[l]["converged_outer"] = out.converged
        log[l]["GW cost"] = out.reg_gw_cost
        log[l]["time"] = end - start
        log[l]["cost_time"] = cost_time
    return Ts, log


def get_coupling_egw_all_ott(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]], eps: float = 5e-3
) -> Tuple[Dict[Number, np.array], Dict]:
    r"""Returns GW coupling between two datasets X, Y, all-to-all manner disregarding labels.

    The function solves the following optimization problem:

    .. math::

        GW = \min_{T\in C_{p,q}} \sum_{i,j,k,l} |(x_i-x_k)^2 - (y_j-y_l)^2|^{2}*T_{i,j}T_{k,l} - \epsilon H(T)


    Parameters
    ----------
    data :
        (source dataset, target dataset) where source and target datasets
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps:
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict :
        Optimal Transport coupling between the samples per label
    log :
        Running log

    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_egw_all_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_egw_all_ott((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    Xs_tot = jnp.array(np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0))
    Xt_tot = jnp.array(np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0))
    source_labels = jnp.array(
        np.concatenate([np.repeat(l, X_dict[l].shape[0]) for l in X_dict.keys()])
    )
    target_labels = jnp.array(
        np.concatenate([np.repeat(l, Y_dict[l].shape[0]) for l in X_dict.keys()])
    )
    start = time.time()

    geom_xx = pointcloud.PointCloud(x=Xs_tot, y=Xs_tot, scale_cost="max_cost")
    geom_yy = pointcloud.PointCloud(x=Xt_tot, y=Xt_tot, scale_cost="max_cost")

    cost_time = time.time() - start
    start = time.time()
    print("running EGWOT with ott")

    prob = quadratic_problem.QuadraticProblem(
        geom_xx,
        geom_yy,
    )

    # Instantiate a jitt'ed Gromov-Wasserstein solver
    solver = gromov_wasserstein.GromovWasserstein(
        epsilon=eps, store_inner_errors=True, max_iterations=1000
    )

    out = solver(prob)

    has_converged = bool(out.linear_convergence[out.n_iters - 1])
    log = {}
    log["n_iters_outer"] = out.n_iters
    log["converged_inner"] = has_converged
    log["converged_outer"] = out.converged
    log["GW cost"] = out.reg_gw_cost
    T = np.array(out.matrix)
    print(f"{out.n_iters} outer iterations were needed.")
    print(f"The last Sinkhorn iteration has converged: {has_converged}")
    print(f"The outer loop of Gromov Wasserstein has converged: {out.converged}")
    print(f"The final regularized GW cost is: {out.reg_gw_cost:.3f}")

    end = time.time()
    print("Done running EGWOT with ott")
    log["time"] = end - start
    log["cost_time"] = cost_time
    return T, log


def get_coupling_eot_ott(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]], eps: float = 5e-3
) -> Tuple[Dict[Number, np.array], Dict]:
    r"""Returns OT coupling between two datasets X, Y given the labels, disregarding label information.

    The function solves the following optimization problem:

    .. math::

        EOT = \min_{T\in C_{p,q}} \sum_{i,j} (x_i-y_j)^2 T_{i,j} - \epsilon H(T)\\

    Parameters
    ----------
    data : 
        (source dataset, target dataset) where source and target datasets 
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps: 
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict : 
        Optimal Transport coupling between the samples per label
    log : 
        Running log
    
    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_eot_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_eot_ott((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    Xs_tot = jnp.array(np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0))
    Xt_tot = jnp.array(np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0))
    source_labels = jnp.array(
        np.concatenate([np.repeat(l, X_dict[l].shape[0]) for l in X_dict.keys()])
    )
    target_labels = jnp.array(
        np.concatenate([np.repeat(l, Y_dict[l].shape[0]) for l in X_dict.keys()])
    )
    start = time.time()

    geom = pointcloud.PointCloud(x=Xs_tot, y=Xt_tot, scale_cost="max_cost")

    cost_time = time.time() - start
    start = time.time()
    print("running LEOT with ott")

    out = linear.solve(geometry.Geometry(cost_matrix=geom.cost_matrix, epsilon=eps))

    log = {}
    log["n_iters_outer"] = out.n_iters
    log["converged"] = out.converged
    log["OT cost"] = out.reg_ot_cost
    T = np.array(out.matrix)
    print(f"{out.n_iters} outer iterations were needed.")
    print(f"The last Sinkhorn iteration has converged: {out.converged}")
    print(f"The final regularized OT cost is: {out.reg_ot_cost:.3f}")

    end = time.time()
    print("Done running EOT with ott")
    log["time"] = end - start
    log["cost_time"] = cost_time

    return T, log


def get_coupling_leot_ott(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]], eps: float = 5e-3
) -> Tuple[Dict[Number, np.array], Dict]:
    r"""Returns OT coupling between two datasets X, Y per label.

    The function solves the following optimization problem:

    .. math::

        EOT^l = \min_{T^l} \sum_{i,j} (x_i-y_j)^2 T^l_{i,j} - \epsilon H(T^l)\\

    Parameters
    ----------
    data : 
        (source dataset, target dataset) where source and target datasets 
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps: 
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict : 
        Optimal Transport coupling between the samples per label
    log : 
        Running log
    
    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_leot_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_leot_ott((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    Xs_tot = jnp.array(np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0))
    Xt_tot = jnp.array(np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0))
    source_labels = jnp.array(
        np.concatenate([np.repeat(l, X_dict[l].shape[0]) for l in X_dict.keys()])
    )
    target_labels = jnp.array(
        np.concatenate([np.repeat(l, Y_dict[l].shape[0]) for l in X_dict.keys()])
    )
    start = time.time()

    geom = pointcloud.PointCloud(x=Xs_tot, y=Xt_tot, scale_cost="max_cost")
    geom = geometry.Geometry(cost_matrix=geom.cost_matrix, epsilon=eps)

    cost_time = time.time() - start
    start = time.time()
    print("running LEOT with ott")
    problem = linear_problem.LinearProblem(
        geom, labels_a=source_labels, labels_b=target_labels
    )
    solver = sinkhorn.Sinkhorn()
    out = solver(problem)

    log = {}
    log["n_iters_outer"] = out.n_iters
    log["converged"] = out.converged
    log["OT cost"] = out.reg_ot_cost
    T = np.array(out.matrix)
    print(f"{out.n_iters} outer iterations were needed.")
    print(f"The last Sinkhorn iteration has converged: {out.converged}")
    print(f"The final regularized OT cost is: {out.reg_ot_cost:.3f}")

    end = time.time()
    print("Done running LEOT with ott")
    log["time"] = end - start
    log["cost_time"] = cost_time
    T_dict = {}
    for l in np.unique(source_labels):
        T_dict[l] = T[np.array(source_labels) == l, :][:, np.array(target_labels) == l]
    return T_dict, log
