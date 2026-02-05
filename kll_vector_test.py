import time
import numpy as np
from datasketches import kll_floats_sketch

a = np.random.rand(200000).astype(np.float32)


def bench_scalar():
    sk = kll_floats_sketch(k=200)
    t0 = time.time()
    for v in a:
        sk.update(float(v))
    return time.time() - t0


def bench_numpy():
    sk = kll_floats_sketch(k=200)
    t0 = time.time()
    sk.update(a)
    return time.time() - t0


def bench_list():
    sk = kll_floats_sketch(k=200)
    t0 = time.time()
    sk.update(a.tolist())
    return time.time() - t0


print("scalar:", bench_scalar())
try:
    print("numpy:", bench_numpy())
except Exception as e:
    print("numpy failed:", e)
try:
    print("list:", bench_list())
except Exception as e:
    print("list failed:", e)
