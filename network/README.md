# NetworkX vs RetworkX vs cuGraph vs IGraph vs Snap

```sh
conda env create -f network/env.yml
conda activate adsb_network
conda install -c rapidsai -c nvidia -c numba -c conda-forge cugraph cudf=21.08 python=3.7 cudatoolkit=11.2
python network/bench.py
```

## Links

* All NetworkX algos: [docs](https://networkx.org/documentation/stable/reference/algorithms/index.html)
* All cuGraph algos: [docs](https://github.com/rapidsai/cugraph#currently-supported-features)

## Other Benchmarks

* https://www.timlrx.com/blog/benchmark-of-popular-graph-network-packages-v2