# In this comparison we should compare:
#   1. naive single-threaded NumPy
#   2. MKL-accelerated NumPy
#   3. CUDA-accelerated CuPy
#   4. CUDA and Tensor Cores-accelerated CuPy
#
# Workloads:
#   * generating big random matrix
#   * multiplying big matrices
#   * moving average computation
#   * generating big random matrix
#
# Difference between CuPy and NumPy:
# https://docs.cupy.dev/en/stable/user_guide/difference.html

import os
import time
from dataclasses import dataclass
from typing import Generator

import pandas as pd

from shared import Bench


numpy = None
obj = None
enabled_backend = ''


def run_all_benchmarks(obj) -> Generator[Bench, None, None]:
    sizes = [512 * 2**i for i in range(0, 6)]
    funcs = [
        obj.matrix_multiply, obj.moving_average,
        obj.pearson_correlations, obj.fft2d, obj.singular_decomposition,
        obj.flat_median, obj.flat_sort, obj.flat_sum
    ]

    for func in funcs:
        for size in sizes:
            s = Sample()
            s.operation = func.__name__
            s.size = size

            mat = obj.generate_random_matrix(size)
            start = time.time()
            while True:
                func(mat)
                if obj.backend != numpy:
                    obj.backend.cuda.stream.get_current_stream().synchronize()
                s.iterations += 1
                s.seconds = time.time() - start
                if s.seconds > max_seconds:
                    break

            print(s)
            yield s


def main(cuda_device: int = -1, filename: os.PathLike = 'benchmark.json'):
    # Swap the backend, if GPU is selected
    cuda_device = int(cuda_device)

    if cuda_device >= 0:
        obj = NuMatrix()
        devices = obj.backend.cuda.runtime.getDeviceCount()
        assert devices > 0, "No CUDA-powered device found"
        print('Found {} CUDA devices'.format(devices))

        obj.backend.cuda.runtime.setDevice(cuda_device)
        specs = obj.backend.cuda.runtime.getDeviceProperties(cuda_device)
        name = specs['name'].decode()
        print('Will run on: {}'.format(name))

    else:
        obj = CuMatrix()
        # obj.backend.distutils.system_info as sysinfo
        #libs = set(sysinfo.get_info('blas')['libraries'])
        #print('Using Numpy with BLAS versions:', *libs)
    #dtype = backend.float32

    samples = list(run_all_benchmarks(obj))
    samples = [s.__dict__ for s in samples]
    samples = pd.DataFrame(samples)
    samples['backend'] = obj.backend.__name__

    # Merge with older results, if present
    if os.path.exists(filename):
        old_results = pd.read_json(filename, orient='records')
        samples = pd.concat([old_results, samples], ignore_index=True)
    samples.to_json(filename, orient='records')


if __name__ == '__main__':
    main()
