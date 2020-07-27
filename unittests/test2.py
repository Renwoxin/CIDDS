from sklearn.feature_extraction import FeatureHasher
from sources.preprocess.features import *
from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.models import mechine_learning

import logging

def hash_features(basic_features):
    """

    Args:
        basic_features:

    Returns:

    """

    h = FeatureHasher(n_features=20, input_type='string', dtype=int, alternate_sign=False)
    features_ = [str(basic_features.values.tolist()[i]) for i in range(len(basic_features.values.tolist()))]
    features_ = str(features_)
    features = h.fit_transform(features_)

    return features



if __name__=='__main__':
    h = FeatureHasher(n_features=20, input_type='string', dtype=int, alternate_sign=False)
    a = [[1], [9], [23], [16], [35], [36], [27], [50], [45], [60]]
    b_ = [str(a[i]) for i in range(len(a))]
    b = h.fit_transform(b_)
    print(b_)
    print(b)