import numpy as np



def sample_idx_noreplace(nelt, nsamp):
    for dtype in [np.uint16, np.uint32, np.uint64]:
        if nelt < np.iinfo(dtype).max:
            break
    idx = np.arange(nelt, dtype=dtype)
    idx = np.random.choice(idx, nsamp, replace=False)
    return idx

def sample_idx_replace(nelt, nsamp):
    idx = np.random.randint(0, nelt, size=nsamp)
    return idx

def sample_from_generator(gen, nelt, nsamp, replace=False):
    if replace:
        idx = sample_idx_replace(nelt, nsamp)
    else:
        idx = sample_idx_noreplace(nelt, nsamp)
    idx = set(idx)
    samples = []
    for i, elt in enumerate(gen):
        if i in idx:
            samples.append(elt)
    return samples