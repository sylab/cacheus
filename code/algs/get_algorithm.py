from .min import MIN
from .lru import LRU
from .lfu import LFU
from .mru import MRU
from .arc import ARC
from .lecar import LeCaR
from .alecar6 import ALeCaR6
from .lirs import LIRS
from .dlirs import DLIRS
from .cacheus import Cacheus
from .arcalecar import ARCALeCaR
from .lirsalecar import LIRSALeCaR


def get_algorithm(alg_name):
    alg_name = alg_name.lower()

    if alg_name == 'min':
        return MIN
    if alg_name == 'lru':
        return LRU
    if alg_name == 'lfu':
        return LFU
    if alg_name == 'mru':
        return MRU
    if alg_name == 'arc':
        return ARC
    if alg_name == 'lecar':
        return LeCaR
    if alg_name == 'alecar6':
        return ALeCaR6
    if alg_name == 'lirs':
        return LIRS
    if alg_name == 'dlirs':
        return DLIRS
    if alg_name == 'cacheus':
        return Cacheus
    if alg_name == 'arcalecar':
        return ARCALeCaR
    if alg_name == 'lirsalecar':
        return LIRSALeCaR
    return None
