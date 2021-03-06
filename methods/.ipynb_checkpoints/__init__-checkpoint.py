from .source_only import run_iter_source_only
from .fullt import run_iter_fullt
from .instapbm import run_iter_instapbm
from .moco import run_iter_moco
from .mimcontra import run_iter_mimcontra
from .adr import run_iter_adr
from .cdan import run_iter_cdan
from .cdane import run_iter_cdane
from .dann import run_iter_dann
from .lirr import run_iter_lirr
from .lirrv1 import run_iter_lirrv1
from .irm import run_iter_irm
from .mme import run_iter_mme
from .counting_mim import run_iter_counting_mim

train_method = {
    'source_only':run_iter_source_only,
    'instapbm': run_iter_instapbm,
    'moco': run_iter_moco,
    'mimcontra': run_iter_mimcontra,
    'cdan': run_iter_cdan,
    'cdane': run_iter_cdane,
    'dann': run_iter_dann,
    'adr': run_iter_adr,
    'lirr': run_iter_lirr,
    'lirrv1': run_iter_lirrv1,
    'irm': run_iter_irm,
    'mme': run_iter_mme,
    'fullt': run_iter_fullt,
    'counting_mim': run_iter_counting_mim
}