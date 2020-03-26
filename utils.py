import torch
import numpy as np

EPS = np.finfo(np.float32).eps.item()

import time

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

def normalize(X):
    # Batch
    if isinstance(X, list):
        for i, x in enumerate(X):
            X[i] = _compute_norm(x)
        return X
    # No batch
    else:
        return _compute_norm(X)

def _compute_norm(x):
    mu = x.mean()
    sigma = x.std()
    return (x - mu) / (sigma + EPS)
