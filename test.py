from time import time_ns
import numpar as nw
import numpy as np
import random as rd

rd.seed(6380496)
print("Starting...")


def test_function(name, np_ver, nw_ver, diff_computer):
    print()
    print(f"testing: {name}")

    t1 = time_ns()
    np_out = np_ver()
    t2 = time_ns()
    print("np done", end="\t")
    t3 = time_ns()
    nw_out = nw_ver()
    t4 = time_ns()
    print("nw done")
    total_diff = diff_computer(np_out, nw_out)

    print(f"diff: {total_diff}")
    print(f"np time: {(t2-t1)/1e9}, nw time: {(t4-t3)/1e9}")
    print(f"improvement: {(t2-t1)/(t4-t3)}")


# dot product
N = 1_000_000
xs = [rd.random() * rd.randint(1, 10) for _ in range(N)]
ys = [rd.random() * rd.randint(1, 10) for _ in range(N)]
test_function("dot",
              lambda: np.dot(xs, ys),
              lambda: nw.dot(xs, ys),
              lambda o1, o2: abs(o2-o1))

# norm
test_function("norm",
              lambda: np.linalg.norm(xs),
              lambda: nw.norm(xs),
              lambda o1, o2: abs(o2-o1))

# outer product
N = 10_000
xs = [rd.random() * rd.randint(1, 10) for _ in range(N)]
ys = [rd.random() * rd.randint(1, 10) for _ in range(N)]
test_function("outer",
              lambda: np.outer(xs, ys),
              lambda: nw.outer(xs, ys),
              lambda o1, o2: np.linalg.norm(o1-o2))

# trace
N = 500
A = [[rd.random() * rd.randint(1, 5) for _ in range(N)] for _ in range(N)]
test_function("trace",
              lambda: np.trace(A),
              lambda: nw.trace(A),
              lambda o1, o2: abs(o2-o1))

# det
test_function("det",
              lambda: np.linalg.det(A),
              lambda: nw.det(A),
              lambda o1, o2: abs(o2-o1))

# transpose
# N = 5000
# M = 3000
# A = [[rd.random() * rd.randint(1, 10) for _ in range(N)] for _ in range(M)]
test_function("transpose",
              lambda: np.transpose(A),
              lambda: nw.transpose(A),
              lambda o1, o2: np.linalg.norm(o1-o2))

# matmul
B = [[rd.random() - 0.5 for _ in range(N)] for _ in range(N)]
test_function("matmul",
              lambda: np.matmul(A, B),
              lambda: nw.matmul(A, B),
              lambda o1, o2: np.linalg.norm(o1-o2))

# mat_pow
E = 50
test_function("matrix_power",
              lambda: np.linalg.matrix_power(B, E),
              lambda: nw.matrix_power(B, E),
              lambda o1, o2: np.linalg.norm(o1-o2))

# inv
test_function("inv",
              lambda: np.linalg.inv(A),
              lambda: nw.inv(A),
              lambda o1, o2: np.linalg.norm(o1-o2))

# solve
b = [rd.random() * rd.randint(1, 10) for _ in range(N)]
test_function("solve",
              lambda: np.linalg.solve(A, b),
              lambda: nw.solve(A, b),
              lambda o1, o2: np.linalg.norm(o1-o2))

# matrix_rank
test_function("matrix_rank",
              lambda: np.linalg.matrix_rank(A),
              lambda: nw.matrix_rank(A),
              lambda o1, o2: np.linalg.norm(o1-o2))
