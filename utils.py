import numpy as np
import time
from gym.spaces import Box, Discrete

# Custom utils

def printISO8601():
    '''
    Prints date and time in ISO8601 format
    e.g. 6th June 2018, 11:15 pm, at UTC:
         2018-06-06T23:15:00Z
    The same moment with Central European Timezone offset:
         2018-06-07T00:15:00+01:00
    '''
    t = time.localtime(time.time())
    yr = str(t[0]).zfill(4)
    mth = str(t[1]).zfill(2)
    day = str(t[2]).zfill(2)
    hr = str(t[3]).zfill(2)
    min = str(t[4]).zfill(2)
    sec = str(t[5]).zfill(2)
    if time.timezone == 0:
        tz = 'Z'
    elif time.timezone > 0:
        tz = '+' + str(time.timezone).zfill(2) + ':00'
    elif time.timezone < 0:
        tz = '-' + str(time.timezone).zfill(2) + ':00'
    ISO8601 = yr + '-' + mth + '-' + day + 'T' + hr + ':' + min + ':' + sec + tz
    return ISO8601

def tic():
    return time.time()

def toc(tic):
    print(f"Time: {time.time() - tic}")

# Core utils

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# TRPO

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))

def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.compat.v1.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.compat.v1.assign(p, p_new) for p, p_new in zip(params, new_params)])
