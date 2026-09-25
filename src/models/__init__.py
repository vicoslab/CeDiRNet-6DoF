from models.FPN import FPN, MyFPN
from models.center_estimator import CenterEstimator
from models.center_estimator_fast import CenterEstimatorFast
from models.center_estimator_with_orientation import CenterOrientationEstimator, CenterOrientationEstimatorFast
from models.center_augmentator import CenterAugmentator

def get_model(name, model_opts):
    if name == "fpn":
        model = FPN(**model_opts)
    elif name == "my-fpn":
        model = MyFPN(**model_opts)
    else:
        raise RuntimeError("model \"{}\" not available".format(name))

    return model

def get_center_model(name, model_opts, is_learnable, use_fast_estimator=False):
    if name.lower() == 'spatialembedding':
        pass
    elif name in ['CenterEstimatorOrientation','CenterOrientationEstimator']:
        if use_fast_estimator:
            return CenterOrientationEstimatorFast(model_opts, is_learnable=is_learnable)
        else:
            return CenterOrientationEstimator(model_opts, is_learnable=is_learnable)
    elif name == 'CenterEstimatorFast':
        return CenterEstimatorFast(model_opts, is_learnable=is_learnable)
    else: # PolarVotingCentersMultiscale or CenterEstimator
        if use_fast_estimator:
            return CenterEstimatorFast(model_opts, is_learnable=is_learnable)
        else:
            return CenterEstimator(model_opts, is_learnable=is_learnable)

def get_center_augmentator(name, model_opts):
    return CenterAugmentator(model_opts)
