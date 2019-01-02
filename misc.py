import tensorflow as tf

def get_rank(inputs):
    return len(inputs.get_shape())

def get_xy_axis(ndim):
    if ndim == 4:
        yaxis = 1
        xaxis = 2
    elif ndim == 3:
        yaxis = 0
        xaxis = 1
    else:
        raise ValueError('Input tensor must be 4D or 3D tensor')
    return xaxis, yaxis    