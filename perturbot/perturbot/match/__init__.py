from .cot_labels import get_coupling_cotl, get_coupling_cotl_sinkhorn, cotl_numpy
from .ott_egwl import (
    get_coupling_egw_labels_ott,
    get_coupling_egw_all_ott,
    get_coupling_eot_ott,
    get_coupling_leot_ott,
    get_coupling_egw_ott,
)
from .cot import get_coupling_cot, get_coupling_cot_sinkhorn, cot_numpy
from .gw_labels import get_coupling_gw_labels
from .fot import get_coupling_fot

__all__ = [
    "cotl_numpy",
    "cot_numpy",
    "get_coupling_eot_ott",
    "get_coupling_leot_ott",
    "get_coupling_cot",
    "get_coupling_cot_sinkhorn",
    "get_coupling_cotl",
    "get_coupling_cotl_sinkhorn",
    "get_coupling_egw_ott",
    "get_coupling_gw_labels",
    "get_coupling_egw_all_ott",
    "get_coupling_egw_labels_ott",
    "get_coupling_fot",
]
