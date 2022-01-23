#!/usr/bin/env python3

from datetime import datetime
from functools import wraps
from typing import List
import time

import pandas

import pretty_print as pp


class FuncStats:
    def __init__(self, name):
        counter_type = float
        self.elapsed_secs = counter_type(0.0)
        self.count_calls = counter_type(0)
        self.name = name

    def __repr__(self) -> str:
        calls_str = pp.integer(self.calls())
        perf_str = pp.seconds(self.mean_seconds())
        duration_str = pp.seconds(self.total_seconds())
        return f'{perf_str} per run (x {calls_str} calls = {duration_str})'

    def log_seconds(self, elapsed_secs: float, items_within_call: int = 1):
        self.elapsed_secs += elapsed_secs
        self.count_calls += items_within_call

    def log_callable(self, callable, *vargs, **kwargs):
        ts = datetime.now()
        result = callable(*vargs, **kwargs)
        te = datetime.now()
        t_diff = te-ts
        items = kwargs.pop('items_within_call', 1)
        self.log_seconds(t_diff.total_seconds(), items_within_call=items)
        return result

    def calls(self) -> int:
        return int(self.count_calls)

    def total_seconds(self) -> float:
        return float(self.elapsed_secs)

    def mean_seconds(self) -> float:
        cnt = self.calls()
        if cnt == 0:
            return 0
        return self.total_seconds() / cnt


class StatsRepo:
    """
        Centralized stats aggregation location
    """

    __shared = None

    @staticmethod
    def shared():
        """ Static access method. """
        if StatsRepo.__shared == None:
            StatsRepo.__shared = StatsRepo([])
        return StatsRepo.__shared

    def __init__(self, func_stats: List[FuncStats]):
        self.stats = dict()
        for fs in func_stats:
            self.stats[fs.name] = fs

    def log_callable(self, name: str, callable, *vargs, **kwargs):
        s = self.stats.get(name, FuncStats(name))
        result = s.log_callable(callable, *vargs, **kwargs)
        self.stats[name] = s
        return result

    def table(self) -> pandas.DataFrame:
        stats_list = list(self.stats.values())
        dur = sum([x.total_seconds() for x in stats_list])

        columns = {}
        columns['name'] = [x.name for x in stats_list]
        columns['calls'] = [x.calls() for x in stats_list]

        if dur > 0:
            columns['runtime_share'] = [
                x.total_seconds() * 100.0 / dur for x in stats_list]
            columns['total_duration'] = [pp.seconds(
                x.total_seconds()) for x in stats_list]
            columns['mean_duration'] = [pp.seconds(
                x.mean_seconds()) for x in stats_list]

        return pandas.DataFrame(columns)

    def _repr_html_(self):
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html
        return self.table().to_html()

    def __repr__(self) -> str:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_string.html
        return self.table().to_string()


def timing(f):
    @wraps(f)
    def wrap(*vargs, **kwargs):
        return StatsRepo.shared().log_callable(f.__name__, f, *vargs, **kwargs)
    return wrap


class Timer:

    def __init__(self, seconds_from_now: float) -> None:
        self.timeout = time.time() + seconds_from_now

    def expired(self) -> bool:
        return self.timeout < time.time()
