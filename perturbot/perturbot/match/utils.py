import numpy as np
from scipy import stats
from scipy.sparse import random


def sinkhorn_scaling(
    a,
    b,
    K,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    always_raise=False,
    **kwargs,
):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    # print(reg)
    # print(np.min(K))

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while err > stopThr and cpt < numItermax:
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1.0 / np.dot(Kp, v)

        zero_in_transp = np.any(KtransposeU == 0)
        nan_in_dual = np.any(np.isnan(u)) or np.any(np.isnan(v))
        inf_in_dual = np.any(np.isinf(u)) or np.any(np.isinf(v))
        if zero_in_transp or nan_in_dual or inf_in_dual:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print("Warning: numerical errors at iteration in sinkhorn_scaling", cpt)
            # if zero_in_transp:
            # print('Zero in transp : ',KtransposeU)
            # if nan_in_dual:
            # print('Nan in dual')
            # print('u : ',u)
            # print('v : ',v)
            # print('KtransposeU ',KtransposeU)
            # print('K ',K)
            # print('M ',M)

            #    if always_raise:
            #        raise NanInDualError
            # if inf_in_dual:
            #    print('Inf in dual')
            u = uprev
            v = vprev

            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum(
                    (v - vprev) ** 2
                ) / np.sum((v) ** 2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b)) ** 2
            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))
        cpt = cpt + 1
    if log:
        log["u"] = u
        log["v"] = v

    if nbb:  # return only loss
        res = np.zeros((nbb))
        for i in range(nbb):
            res[i] = np.sum(u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def random_gamma_init(p, q, **kwargs):
    """Returns random coupling matrix with marginal p,q"""
    rvs = stats.beta(1e-1, 1e-1).rvs
    S = random(len(p), len(q), density=1, data_rvs=rvs)
    return sinkhorn_scaling(p, q, S.A, **kwargs)


def init_matrix_np(X1, X2, v1, v2):
    """Return loss matrices and tensors for COOT fast computation
    Returns the value of |X1-X2|^{2} \otimes T as done in [1] based on [2] for the Gromov-Wasserstein distance.
    Where :
        - X1 : The source dataset of shape (n,d)
        - X2 : The target dataset of shape (n',d')
        - v1 ,v2 : weights (histograms) on the columns of resp. X1 and X2
        - T : Coupling matrix of shape (n,n')
    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2.

    Returns
    -------
    constC : ndarray, shape (n, n')
        Constant C matrix (see paragraph 1.2 of supplementary material in [1])
    hC1 : ndarray, shape (n, d)
        h1(X1) matrix (see paragraph 1.2 of supplementary material in [1])
    hC2 : ndarray, shape (n', d')
        h2(X2) matrix (see paragraph 1.2 of supplementary material in [1])
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    .. [2] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    def f1(a):
        return a**2

    def f2(b):
        return b**2

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = np.dot(
        np.dot(f1(X1), v1.reshape(-1, 1)), np.ones(f1(X2).shape[0]).reshape(1, -1)
    )
    constC2 = np.dot(
        np.ones(f1(X1).shape[0]).reshape(-1, 1), np.dot(v2.reshape(1, -1), f2(X2).T)
    )

    constC = constC1 + constC2
    hX1 = h1(X1)
    hX2 = h2(X2)

    return constC, hX1, hX2
