# NetworkX vs RetworkX vs cuGraph vs IGraph vs Snap

```sh
conda env create -f network/env.yml
conda activate adsb_network
conda install -c rapidsai -c nvidia -c conda-forge cugraph=22.08 python=3.9 cudatoolkit=11.5
python network/bench.py
```

## Links

* All NetworkX algos: [docs](https://networkx.org/documentation/stable/reference/algorithms/index.html)
* All cuGraph algos: [docs](https://github.com/rapidsai/cugraph#currently-supported-features)

## Other Benchmarks

* https://www.timlrx.com/blog/benchmark-of-popular-graph-network-packages-v2