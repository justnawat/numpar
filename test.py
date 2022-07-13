from time import time_ns
import numpar as nw
import numpy as np
import random as rd

rd.seed(6380496)


def test_function(name, np_ver, nw_ver, diff_computer):
    t1 = time_ns()
    np_out = np_ver()
    t2 = time_ns()
    nw_out = nw_ver()
    t3 = time_ns()
    total_diff = diff_computer(np_out, nw_out)

    print(f"testing: {name}")
    print(f"diff: {total_diff}")
    print(f"np time: {(t2-t1)/1e9}, nw time: {(t3-t2)/1e9}")
    print(f"improvement: {(t2-t1)/(t3-t2)}")
    print()


# dot product
N = 10_000_000
xs = [rd.random() * rd.randint(1, 10) for _ in range(N)]
ys = [rd.random() * rd.randint(1, 10) for _ in range(N)]
test_function("dot",
              lambda: np.dot(xs, ys),
              lambda: nw.dot(xs, ys),
              lambda o1, o2: abs(o2-o1))

# outer product
N = 10_000
xs = [rd.random() * rd.randint(1, 10) for _ in range(N)]
ys = [rd.random() * rd.randint(1, 10) for _ in range(N)]
test_function("outer",
              lambda: np.outer(xs, ys),
              lambda: nw.outer(xs, ys),
              lambda o1, o2: sum([sum(abs(np.array(p) - np.array(w))) for p, w in zip(o1, o2)]))

# trace
N = 10_000
A = [[rd.random() * rd.randint(1, 10) for _ in range(N)] for _ in range(N)]
test_function("trace",
              lambda: np.trace(A),
              lambda: nw.trace(A),
              lambda o1, o2: abs(o2-o1))

# norm
N = 10_000_000
xs = [rd.random() * rd.randint(1, 10) for _ in range(N)]
test_function("norm",
              lambda: np.linalg.norm(xs),
              lambda: nw.norm(xs),
              lambda o1, o2: abs(o2-o1))

# transpose
N = 5000
M = 3000
A = [[rd.random() * rd.randint(1, 10) for _ in range(N)] for _ in range(M)]
test_function("transpose",
              lambda: np.transpose(A),
              lambda: nw.transpose(A),
              lambda o1, o2: sum([sum(abs(np.array(p) - np.array(w))) for p, w in zip(o1, o2)]))
