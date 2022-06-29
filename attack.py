import numpy as np

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

def global_minl2(A, x):
    print(A.shape, x.shape)
    b = np.dot(A, x)
    print(b.shape)
    b = (b > 0.5).astype(float)
    b = b[:, np.sum(b, axis=0) > 0]
    if b.size == 0:
        return None, 1e30
    x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
    print(x.shape)
    b = np.dot(A, x)
    l2 = np.sum(np.minimum(b ** 2, (b - 1.) ** 2), axis=0)
    pos = np.argmin(l2)
    return x[:, pos], l2[pos]
    
def leverage_score_solve(A, it, k):
    sol, val = global_minl2(A, np.ones((A.shape[1], 1)))
    # run several iterations
    for i in range(it):
        x = leverage_score_sampling(A, k)
        p, v = global_minl2(A, x)
        if v < val:
            sol, val = p, v
    return sol, val
