import math
from statistics import mean
from typing import List, Dict, Tuple


from tqdm import tqdm

import shared


class ViaAnyBatched:
    """
        Splits big dataset processing into multi-file batches.
        It allows us to fit into VRAM on smaller GPUs.
    """

    def __init__(self, engine, files_per_batch: int = 10) -> None:
        self.engine = engine
        self.files_per_batch = files_per_batch

    def load(self, df_or_paths):
        self.df_or_paths = df_or_paths

    def query1(self) -> Dict[str, int]:
        accumulated = {}
        progress = tqdm(range(self._count_batches()))
        progress.set_description('Query 1')
        for batch_idx in progress:
            batch = self._prepare_batch(batch_idx)
            self.engine.load(batch)
            addition = self.engine.query1()
            accumulated = shared.dicts_union(accumulated, addition, sum)
        return accumulated

    def query2(self) -> Dict[int, float]:
        accumulated = {}
        progress = tqdm(range(self._count_batches()))
        progress.set_description('Query 2')
        for batch_idx in progress:
            batch = self._prepare_batch(batch_idx)
            self.engine.load(batch)
            addition = self.engine.query2()
            accumulated = shared.dicts_union(accumulated, addition, mean)
        return accumulated

    def query3(self) -> Dict[Tuple[int, int], int]:
        accumulated = {}
        progress = tqdm(range(self._count_batches()))
        progress.set_description('Query 3')
        for batch_idx in progress:
            batch = self._prepare_batch(batch_idx)
            self.engine.load(batch)
            addition = self.engine.query3()
            accumulated = shared.dicts_union(accumulated, addition, sum)
        return accumulated

    def query4(self) -> List[Tuple[int, int, int, int]]:
        accumulated = {}
        progress = tqdm(range(self._count_batches()))
        progress.set_description('Query 4')
        for batch_idx in progress:
            batch = self._prepare_batch(batch_idx)
            self.engine.load(batch)
            addition = self.engine.query4()
            addition = {t[:3]: t[3] for t in addition}
            accumulated = shared.dicts_union(accumulated, addition, sum)

        # Export everything back into a list of tuples and sort them
        accumulated = [k + (v, ) for k, v in accumulated.items()]
        accumulated = sorted(accumulated, key=lambda t: t[3], reverse=True)
        return accumulated

    def memory_usage(self) -> int:
        return self.engine.memory_usage()

    def close(self):
        self.engine.close()

    def _count_batches(self) -> int:
        if isinstance(self.df_or_paths, list):
            return int(math.ceil(len(self.df_or_paths) / self.files_per_batch))
        else:
            return 1

    def _prepare_batch(self, i: int):
        if isinstance(self.df_or_paths, list):
            return self.df_or_paths[i*self.files_per_batch:][:self.files_per_batch]
        else:
            return self.df_or_paths
