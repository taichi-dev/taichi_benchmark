import time
import taichi as ti
import numpy as np
from numba import cuda

lx, ly, nx, ny = 1.0, 1.0, 4096, 4096
dx, dy, dt = lx/nx, ly/ny, 0.001
mu = 0.001


def gen_data_numpy():
    np.random.seed(42)
    u = np.random.rand(nx + 1, ny + 2).astype(np.float32)
    v = np.random.rand(nx + 2, ny + 1).astype(np.float32)
    ut = np.ones((nx + 1, ny + 2), dtype=np.float32)
    vt = np.ones((nx + 2, ny + 1), dtype=np.float32)
    return u, v, ut, vt


def gen_data_numba_cuda():
    u, v, ut, vt = gen_data_numpy()
    u_dev = cuda.to_device(u)
    v_dev = cuda.to_device(v)
    ut_dev = cuda.to_device(ut)
    vt_dev = cuda.to_device(vt)
    return u_dev, v_dev, ut_dev, vt_dev


def gen_data_taichi():
    u, v, ut, vt = gen_data_numpy()
    ti_u = ti.ndarray(shape=(nx + 1, ny + 2), dtype=ti.f32)
    ti_v = ti.ndarray(shape=(nx + 2, ny + 1), dtype=ti.f32)
    ti_ut = ti.ndarray(shape=(nx + 1, ny + 2), dtype=ti.f32)
    ti_vt = ti.ndarray(shape=(nx + 2, ny + 1), dtype=ti.f32)
    ti_u.from_numpy(u)
    ti_v.from_numpy(v)
    ti_ut.from_numpy(ut)
    ti_vt.from_numpy(vt)
    return ti_u, ti_v, ti_ut, ti_vt


def benchmark(func):
    def wrapper(u, v, ut, vt, name="Taichi", ut_gt=None, vt_gt=None, nIter=1000):
        func(u, v, ut, vt)
        if ut_gt is not None and vt_gt is not None:
            if type(ut) is ti.lang._ndarray.ScalarNdarray:
                ut_diff = ut.to_numpy()
                vt_diff = vt.to_numpy()
            elif type(ut) is cuda.cudadrv.devicearray.DeviceNDArray:
                ut_diff = ut.copy_to_host()
                vt_diff = vt.copy_to_host()
            else:
                ut_diff = ut
                vt_diff = vt
            print(f"Verify UT matrix for {name}: ", np.allclose(
                ut_diff, ut_gt, atol=1, rtol=1, equal_nan=True))
            print(f"Verify VT matrix for {name}: ", np.allclose(
                vt_diff, vt_gt, atol=1, rtol=1, equal_nan=True))
        now = time.perf_counter()
        for _ in range(nIter):
            func(u, v, ut, vt)
        end = time.perf_counter()
        print(f'Time spent in {name} for {nIter}x: {(end -now):4.2f} sec.')
    return wrapper


def compute_ground_truth(calc_velocity_func):
    u, v, ut, vt = gen_data_numpy()
    # Numpy is way too slow even for ground truth computation.
    calc_velocity_func(u, v, ut, vt)
    return ut, vt
