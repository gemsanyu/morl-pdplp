import numpy as np
import numba as nb

@nb.guvectorize(['void(float32[:],float32[:],float32[:])'], '(n),(n)->(n)')
def cek(A,B,C):
    for i in range(len(A)):
        temp = A[i]
        curr = temp
        curr = temp + B[i]/2
        C[i] = curr - temp

if __name__ == '__main__':
    A = np.zeros(5, dtype=np.float32)
    B = np.arange(5, dtype=np.float32)
    C = np.empty(5, dtype=np.float32)
    cek(A,B,C)
    print(A)
    print(B)
    print(C)