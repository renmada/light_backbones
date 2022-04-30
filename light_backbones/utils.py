from paddle.nn import BatchNorm1D, BatchNorm2D, LayerNorm, Identity
from paddle.nn import ReLU, Hardswish, Hardsigmoid, PReLU, LeakyReLU, Swish, GELU, ReLU6, Sigmoid


def get_normalization_layer(num_features, norm_type='batch_norm_2d'):
    norm_type = norm_type.lower()
    if norm_type in ['batch_norm', 'batch_norm_2d']:
        norm_layer = BatchNorm2D(num_features)
    elif norm_type == 'batch_norm_1d':
        norm_layer = BatchNorm1D(num_features)
    elif norm_type in ['layer_norm', 'ln']:
        norm_layer = LayerNorm(num_features)
    return norm_layer


def get_activation_fn(act_type='swish', num_parameters=-1,
                      negative_slope=0.1):
    if act_type == 'relu':
        return ReLU()
    elif act_type == 'prelu':
        assert num_parameters >= 1
        return PReLU(num_parameters=num_parameters)
    elif act_type == 'leaky_relu':
        return LeakyReLU(negative_slope=negative_slope)
    elif act_type == 'hard_sigmoid':
        return Hardsigmoid()
    elif act_type == 'swish':
        return Swish()
    elif act_type == 'gelu':
        return GELU()
    elif act_type == 'sigmoid':
        return Sigmoid()
    elif act_type == 'relu6':
        return ReLU6()
    elif act_type == 'hard_swish':
        return Hardswish()


def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


