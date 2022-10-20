import numpy as np

def qr(a : np.ndarray) :
    m, n = a.shape
    q = np.empty((m, 0))
    i = 0
    while q.shape[1] < m:
        ai = a[:, i] if i < n else np.random.rand(m)
        bi = ai
        for j in range(q.shape[1]):
            qj = q[:, j]
            proj = np.dot(ai, qj) * qj
            bi = bi - proj
        if not np.allclose(np.zeros_like(bi), bi):
            ei = bi / np.linalg.norm(bi)
            q = np.hstack((q, ei.reshape(-1, 1)))
        i += 1

    r = np.dot(q.T, a)
    return q, r

print(qr(np.array([[1, 2, 4], [3, 3, 2], [4, 1, 3]])))
