from .random_sampling import RandomSampling
from .kmeans_sampling import KMeansSampling
from .kcenter_greedy import KCenterGreedy, KCenterGreedy2, CdalCS
from .bald import BALDDropout
from .badge_sampling import BadgeSampling
from .falcun import Falcun
from .zhdanov import ZhdanovWeightedKMeans
from .kmeanspp import KMeansPP, KMeansPPSampling
from .clue import CLUE
from .uncertainty import MarginSampling, EntropySampling, LeastConfidence, MarginWeighted, EntropyWeighted, LeastConfidenceWeighted
from .alpha_mix import AlphaMixSampling
from .falcun import Falcun


def get_strategy(name, x, y, labelled, model, d_handler, d_args, strat_args):
    klass = globals()[name]
    instance = klass(x, y, labelled, model, d_handler, d_args, **strat_args)
    return instance
