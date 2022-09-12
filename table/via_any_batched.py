import math
from statistics import mean
from typing import List, Dict, Tuple

from tqdm import tqdm
import pyarrow as pa

import shared
import dataset


class ViaAnyBatched:
    """
        Splits big dataset processing into multi-file batches.
        It allows us to fit into VRAM on smaller GPUs.
    """

    def __init__(self, engine, files_per_batch: int = 10, prefetch: bool = False) -> None:
        self.engine = engine
        self.files_per_batch = files_per_batch
        self.prefetch = prefetch

    def load(self, df_or_paths):
        if not self.prefetch:
            self.df_or_paths = df_or_paths
            self.dfs = None
        else:
            self.df_or_paths = df_or_paths
            self.dfs = [dataset.read_parquet_dataset(
                path) for path in df_or_paths]

    def query1(self) -> Dict[str, int]:
        accumulated = {}
        progress = tqdm(
            range(0, self._count_files(), self.files_per_batch),
            desc='Query 1',
            unit='batch'
        )
        for first_file_idx in progress:
            batch = self._prepare_batch(first_file_idx)
            self.engine.load(batch)
            addition = self.engine.query1()
            self.engine.close()
            accumulated = shared.dicts_union(accumulated, addition, sum)
        return accumulated

    def query2(self) -> Dict[int, float]:
        accumulated = {}
        progress = tqdm(
            range(0, self._count_files(), self.files_per_batch),
            desc='Query 2',
            unit='batch'
        )
        for first_file_idx in progress:
            batch = self._prepare_batch(first_file_idx)
            self.engine.load(batch)
            addition = self.engine.query2()
            self.engine.close()
            accumulated = shared.dicts_union(accumulated, addition, mean)
        return accumulated

    def query3(self) -> Dict[Tuple[int, int], int]:
        accumulated = {}
        progress = tqdm(
            range(0, self._count_files(), self.files_per_batch),
            desc='Query 3',
            unit='batch'
        )
        for first_file_idx in progress:
            batch = self._prepare_batch(first_file_idx)
            self.engine.load(batch)
            addition = self.engine.query3()
            self.engine.close()
            accumulated = shared.dicts_union(accumulated, addition, sum)
        return accumulated

    def query4(self) -> List[Tuple[int, int, int, int]]:
        accumulated = {}
        progress = tqdm(
            range(0, self._count_files(), self.files_per_batch),
            desc='Query 4',
            unit='batch'
        )
        for first_file_idx in progress:
            batch = self._prepare_batch(first_file_idx)
            self.engine.load(batch)
            addition = self.engine.query4()
            self.engine.close()
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

    def _count_files(self) -> int:
        if isinstance(self.df_or_paths, list):
            return len(self.df_or_paths)
        else:
            return 1

    def _count_batches(self) -> int:
        if isinstance(self.df_or_paths, list):
            return int(math.ceil(len(self.df_or_paths) / self.files_per_batch))
        else:
            return 1

    def _prepare_batch(self, first_file_idx: int):
        if isinstance(self.df_or_paths, list):
            if self.dfs:
                slice = self.dfs[first_file_idx:first_file_idx +
                                 self.files_per_batch]
                return pa.concat_tables(slice)
            else:
                return self.df_or_paths[first_file_idx:first_file_idx+self.files_per_batch]
        else:
            return self.df_or_paths
