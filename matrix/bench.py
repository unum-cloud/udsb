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
from typing import Generator, Optional, List

import numpy as np

from shared import Bench, run_persisted_benchmarks


def benchmarks_for_backend(class_: type, class_name: str, size: int) -> Generator[Bench, None, None]:

    obj = None

    def generate():
        nonlocal obj
        obj = class_(side=size)

    funcs = [generate]
    funcs.append([
        lambda obj=obj: obj.matrix_multiply(),
        lambda obj=obj: obj.moving_average(),
        lambda obj=obj: obj.pearson_correlations(),
        lambda obj=obj: obj.fft2d(),
        lambda obj=obj: obj.singular_decomposition(),
        lambda obj=obj: obj.flat_median(),
        lambda obj=obj: obj.flat_sort(),
        lambda obj=obj: obj.flat_sum(),
    ])

    funcs_names = [
        'Generate Random Matrix',
        'Matrix Multiply', 'Rows Moving Average', 'Pearson Correlation of Rows',
        '2D FFT', 'Singular Values Decomposition',
        'Array Median', 'Array Sorting', 'Array Summation',
    ]

    for func, func_name in zip(funcs, funcs_names):
        yield Bench(
            operation=func_name,
            backend=class_name,
            dataset=f'{size}x{size}',
            dataset_bytes=(size ** 2)*4,
            func=func,
        )

    if obj is not None:
        obj.close()
    obj = None


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

        cupy.runtime.setDevice(cuda_device)
        specs = cupy.runtime.getDeviceProperties(cuda_device)
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
    benches = list(available_benchmarks())
    backends = np.unique([x.backend for x in benches])
    datasets = np.unique([x.dataset for x in benches])
    results_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        'report/results.json'
    )

    print('Available backends: ', backends)
    print('Available datasets: ', datasets)

    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO'),
        format='%(asctime)s: %(message)s',
    )
    run_persisted_benchmarks(benches, 10, results_path)
