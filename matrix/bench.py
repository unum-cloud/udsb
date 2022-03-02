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

import logging
import os
import pathlib
from dataclasses import dataclass
from sys import argv
from typing import Generator, Optional, List

import numpy as np

from shared import Bench, run_persisted_benchmarks


def benchmarks_for_backend(class_: type, class_name: str, size: int) -> Generator[Bench, None, None]:

    funcs = [
        ('Generate', lambda: globals().update({'m': class_(side=size)})),
        ('Matrix Multiply', lambda: globals()['m'].matrix_multiply()),
        ('Moving Averages', lambda: globals()['m'].moving_average()),
        ('Pearson Correlation', lambda: globals()['m'].pearson_correlations()),
        ('2D FFT', lambda: globals()['m'].fft2d()),
        ('Matrix SVD', lambda: globals()['m'].singular_decomposition()),
        ('Array Median', lambda: globals()['m'].flat_median()),
        ('Array Sorting', lambda: globals()['m'].flat_sort()),
        ('Array Summation', lambda: globals()['m'].flat_sum()),
    ]

    for func_name, func in funcs:
        yield Bench(
            operation=func_name,
            backend=class_name,
            dataset=f'{size}x{size}',
            dataset_bytes=(size ** 2)*4,
            func=func,
        )


def benchmarks_for_sizes(class_: type, class_name: str, side_sizes: List[int]) -> Generator[Bench, None, None]:
    for size in side_sizes:
        yield from benchmarks_for_backend(class_, class_name, size)


def available_benchmarks(
    cuda_device: int = -1,
    class_name: Optional[str] = None,
    logger: logging.Logger = logging.getLogger(),
) -> Generator[Bench, None, None]:

    # Swap the backend, if GPU is selected
    cuda_device = int(cuda_device)
    sizes = [512 * 2**i for i in range(0, 6)]

    if cuda_device >= 0:
        import cupy
        devices = cupy.cuda.runtime.getDeviceCount()
        assert devices > 0, 'No CUDA-powered device found'
        logger.info('Found {} CUDA devices'.format(devices))

        cupy.cuda.runtime.setDevice(cuda_device)
        specs = cupy.cuda.runtime.getDeviceProperties(cuda_device)
        name = specs['name'].decode()
        logger.info('Will run on: {}'.format(name))

        if class_name is None:
            class_name = 'CuPy'

        from via_cupy import ViaCuPy
        yield from benchmarks_for_sizes(ViaCuPy, class_name, sizes)

    else:
        import numpy.distutils.system_info as sysinfo
        libs = set(sysinfo.get_info('blas')['libraries'])
        libs_str = ','.join(libs)
        logger.info(f'Using NumPy with BLAS versions: {libs_str}')

        if class_name is None:
            class_name = 'NumPy'

        from via_numpy import ViaNumPy
        yield from benchmarks_for_sizes(ViaNumPy, class_name, sizes)


if __name__ == '__main__':
    gpu = argv[1] if len(argv) == 2 else -1
    benches = list(available_benchmarks(gpu))
    backends = np.unique([x.backend for x in benches])
    datasets = np.unique([x.dataset for x in benches])
    results_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        'report/results.json'
    )

    print('Available backends: ', backends)
    print('Available datasets: ', datasets)
    run_persisted_benchmarks(benches, 10, results_path)
