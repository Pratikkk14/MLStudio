from enum import Enum

class NumericalImputationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"

class ScalingStrategy(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"

class CategoricalEncodingStrategy(str, Enum):
    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    NONE = "none"
