# Accelerated DataScience Benchmark

A comparison of most commonly used Data-Science Python packages and their alternatives.
Generally, those alternatives have identical Python interfaces, but come with Multi-Threaded CPU or even GPU backends, implemented in C++, CUDA, Rust and other low-level languages.

## Matrices

For Linear Algebra and Digital Signal Processing we synthetically generate square random matrices, mainly of with single-precision floating point numbers.
That is different from the default Pythons `float` that uses the 64-bit representation, more commonly described as `double` in C-like languages.
Participating packages:

* NumPy over BLIS
* NumPy over OpenBLAS
* NumPy over Intel MKL and One API
* CuPy over CuBLAS

## Graphs or Networks

For Graph Theoretical and Network Science workloads we pick various commonly used datasets from the Stanford Network Repository.
All ranging under 1 MB to over 1 GB and 100 million edges.
Participating packages:

* NetworkX
* RetworkX
* CuGraph

## Tabular Data

We took the NYC Taxi Rides dataset as our primary dataset and run the classical 4-query benchmark on its subsets.
Participating packages:

* Pandas
* Modin
* CuDF
* Dask-CuDF
* SQLite
* Apache DataFusion
