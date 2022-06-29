import numpy as np
from numba import jit

def create_enum(d):
    pw = 2 ** d
    a = np.array(range(pw))
    res = np.zeros((d, pw))
    for i in range(d):
        res[i, ((a >> i) & 1) == 1] = 1
    return res

def leverage_score_sampling(A, k):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    l = np.sum(u ** 2, axis=1)
    p = l / np.sum(l)
    row = np.random.choice(A.shape[0], k, replace=True, p=p)
    p = p.reshape(A.shape[0], 1)
    S = A[row, :] / np.sqrt(k * p[row, :])
    enum = create_enum(k)[:, 1:] # delete all 0
    r = enum / np.sqrt(k * p[row, :])
    sol = np.linalg.solve(np.dot(S.T, S), np.dot(S.T, r))
    return sol

@jit(nopython=True)
def global_minl2(A, x):
    cov = np.dot(A.T, A)
    sol, val = None, None
    for i in range(x.shape[1]):
        b = np.dot(A, x[:, i])
        b = (b > 0.5).astype(np.float32)
        if np.isclose(np.sum(b), 0):
            continue
        y = np.linalg.solve(cov, np.dot(A.T, b))
        b = np.dot(A, y)
        l2 = np.sum(np.minimum(b ** 2, (b - 1.) ** 2))
        if val is None or l2 < val:
            sol, val = y, l2
    return sol, val

def leverage_score_solve(A, it, k):
    A = A.astype(float)
    sol, val = global_minl2(A, np.ones((A.shape[1], 1)))
    # run several iterations
    for i in range(it):
        print(i)
        x = leverage_score_sampling(A, k)
        p, v = global_minl2(A, x)
        if val is None or (v is not None and v < val):
            sol, val = p, v
    return sol, val
