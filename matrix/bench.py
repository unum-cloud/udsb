# In this comparison we should compare:
#   1. naive single-threaded NumPy
#   2. MKL-accelerated NumPy
#   3. CUDA-accelerated CuPy
#   4. CUDA and Tensor Cores-accelerated CuPy
#   5. JAX multi-gpu
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
import numpy.distutils.system_info as sysinfo


from shared import Bench, run_persisted_benchmarks


def benchmarks_for_backend(class_: type, class_name: str, size: int, **kwargs) -> Generator[Bench, None, None]:

    funcs = [
        ('Generate', lambda: globals().update({'m': class_(side=size, **kwargs)})),
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


def benchmarks_for_sizes(class_: type, class_name: str, side_sizes: List[int], **kwargs) -> Generator[Bench, None, None]:
    for size in side_sizes:
        yield from benchmarks_for_backend(class_, class_name, size, **kwargs)


def available_benchmarks(
    cuda_device: int = 0,
    logger: logging.Logger = logging.getLogger(),
) -> Generator[Bench, None, None]:

    # Swap the backend, if GPU is selected
    sizes = [512 * 2**i for i in range(0, 6)]

    try:
        import cupy
        cuda_device = int(cuda_device)
        if 'CUPY_ACCELERATORS' not in os.environ.keys():
            os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"
        if 'CUPY_TF32' not in os.environ.keys():
            os.environ['CUPY_TF32'] = '1'

        devices = cupy.cuda.runtime.getDeviceCount()
        assert devices > 0, 'No CUDA-powered device found'
        logger.info('Found {} CUDA devices'.format(devices))

        cupy.cuda.runtime.setDevice(cuda_device)
        specs = cupy.cuda.runtime.getDeviceProperties(cuda_device)
        name = specs['name'].decode()
        logger.info('Using CuPy with : {}'.format(name))

        from via_cupy import ViaCuPy
        yield from benchmarks_for_sizes(ViaCuPy, 'CuPy', sizes)
    except ModuleNotFoundError:
        logger.info('CuPy not found, skipping')

    try:
        import torch
        logger.info(f'Using Torch with : {torch.cuda.get_device_name()}')

        from via_torch import ViaTorch
        yield from benchmarks_for_sizes(ViaTorch, 'Torch', sizes)
    except ModuleNotFoundError:
        logger.info('Torch not found, skipping')
    
    try:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        import jax
        logger.info(f'Using JAX with : {jax.devices()}')

        from via_jax import ViaJAX
        yield from benchmarks_for_sizes(ViaJAX, f'JAX/{jax.device_count()}', sizes, device_count=jax.device_count())
        yield from benchmarks_for_sizes(ViaJAX, 'JAX/1', sizes, device_count=1)
    except ModuleNotFoundError:
        logger.info('JAX not found, skipping')

    try:
        import numpy as np
        libs = set(sysinfo.get_info('blas')['libraries'])
        libs_str = ','.join(libs)
        logger.info(f'Using NumPy with BLAS versions: {libs_str}')

        from via_numpy import ViaNumPy
        yield from benchmarks_for_sizes(ViaNumPy, 'NumPy', sizes)
    except ModuleNotFoundError:
        logger.info('NumPy not found, skipping')


if __name__ == '__main__':
    benches = list(available_benchmarks())
    backends = {x.backend for x in benches}
    datasets = {x.dataset for x in benches}
    results_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        'report/results.json'
    )

    print('Available backends: ', backends)
    print('Available datasets: ', datasets)
    run_persisted_benchmarks(benches, 10, results_path)
