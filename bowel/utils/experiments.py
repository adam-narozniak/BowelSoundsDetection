import numpy as np


def preprocess_trials(x):
    """Maps empty list to np.nan and strips down one element lists to single values."""
    y = {}
    for k, v in x.items():
        if v:
            y[k] = v[0]
        else:
            y[k] = np.nan
    return y
