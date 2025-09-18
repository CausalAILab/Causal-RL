import numpy as np

def _split_token(tok: str):
    i = 0

    while i < len(tok) and not tok[i].isdigit():
        i += 1

    var = tok[:i]
    t = int(tok[i:])

    return var, t

def one_hot(v: int, n_classes: int): 
    v = int(v)

    if v < 0 or v >= n_classes:
        return np.zeros((n_classes,), dtype=np.float32)
    
    return np.eye(n_classes, dtype=np.float32)[v]

def _width(v: str, obs: dict):
    if v not in obs or len(obs[v]) == 0:
        return 1

    sample = obs[v][0]
    arr = np.array(sample)
    return int(arr.size) if arr.ndim > 0 else 1

def encode_z(obs: dict, t: int, Z_sets: dict, categorical_dims: dict, order='time_name'):
    key = f'X{t}'
    toks = [(_split_token(z)) for z in Z_sets[key]]

    if order == 'time_name':
        toks.sort(key=lambda p: (p[1], p[0]))

    elif order == 'name_time':
        toks.sort(key=lambda p: (p[0], p[1]))

    vals = []
    for var, tt in toks:
        if var not in obs:
            width = categorical_dims.get(var, _width(var, obs))
            vals.append(np.zeros((width,), dtype=np.float32))
            continue           

        history = obs[var]
        if tt < 0 or tt >= len(history):
            width = categorical_dims.get(var, _width(var, obs))
            vals.append(np.zeros((width,), dtype=np.float32))
            continue

        v = history[tt]

        if var in categorical_dims:
            enc = one_hot(v, categorical_dims[var])

        else:
            enc = np.array(np.asarray(v, dtype=np.float32)).reshape(-1)

        vals.append(enc.astype(np.float32, copy=False))

    if not vals:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(vals, axis=0).astype(np.float32, copy=False)