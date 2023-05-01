
def uniform_exclude_inner(np_uniform, a, b, a_i, b_i):
    """Draw sample from uniform distribution, excluding an inner range"""
    if not (a < a_i and b_i < b):
        raise ValueError(
            "Bad range, inner: ({},{}), outer: ({},{})".format(a, b, a_i, b_i)
        )
    lower_sample = np_uniform(a, a_i)
    upper_sample = np_uniform(b_i, b)
    if np_uniform(0,1) < 0.5:
        return lower_sample
    else:
        return upper_sample
    
    # while True:
    #     # Resample until value is in-range
    #     result = np_uniform(a, b)
    #     if (a <= result and result < a_i) or (b_i <= result and result < b):
    #         return result
