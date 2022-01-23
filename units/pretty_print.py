#!/usr/bin/env python3

import datetime as dt
import time


def num(
    value: float,
    decimal_places: int = 2,
    denominator: float = 1000.0,
    units=['', 'K', 'M', 'G', 'T'],
) -> str:

    if value == 0:
        return '0'
    if value < 0:
        return '-' + num(abs(value), decimal_places, denominator, units)
    unit = ''
    for unit in units:
        if value < denominator:
            break
        value /= denominator
    return f'{value:.{decimal_places}f}{unit}'


def percent(value: float, decimal_places: int = 2) -> str:
    return num(value * 100.0, decimal_places, units=[]) + '%'


def integer(value, decimal_places=0) -> str:
    return num(value, decimal_places)


def integer_bytes(value, decimal_places=2) -> str:
    us = [' B', ' KiB', ' MiB', ' GiB', ' TiB']
    return num(value, decimal_places, denominator=1024.0, units=us)


def delta_time(from_seconds, until_seconds=None) -> str:
    """
    delta_time(1, 2)           # 1 second ago
    delta_time(1000, 9000)     # 2 hours, 133 minutes ago
    delta_time(1000, 987650)   # 11 days ago
    delta_time(1000)           # 15049 days ago (relative to now)
    """
    if not until_seconds:
        until_seconds = time.time()

    seconds = until_seconds - from_seconds
    delta = dt.timedelta(seconds=seconds)
    if not until_seconds:
        return milliseconds(delta) + ' ago'
    return milliseconds(delta)


def milliseconds(ms: int) -> str:
    if isinstance(ms, dt.timedelta):
        ms = ms.total_seconds() * 1000

    remainder = int(ms)
    periods = [
        ('year',        60 * 60 * 24 * 365 * 1000),
        ('month',       60 * 60 * 24 * 30 * 1000),
        ('day',         60 * 60 * 24 * 1000),
        ('hour',        60 * 60 * 1000),
        ('minute',      60 * 1000),
        ('sec',         1000),
        ('millisec',    1),
    ]

    strings = []
    for name, divisor in periods:
        if remainder > divisor:
            cnt, remainder = divmod(remainder, divisor)
            has_s = 's' if cnt > 1 else ''
            strings.append('%s %s%s' % (cnt, name, has_s))

    return ', '.join(strings)


def seconds(secs: float) -> str:
    return milliseconds(secs * 1000)


def anything(any) -> str:
    return str(any)


def black(text: str) -> str:
    return '\033[30m' + text + '\033[0m'


def red(text: str) -> str:
    return '\033[31m' + text + '\033[0m'


def green(text: str) -> str:
    return '\033[32m' + text + '\033[0m'


def yellow(text: str) -> str:
    return '\033[33m' + text + '\033[0m'


def blue(text: str) -> str:
    return '\033[34m' + text + '\033[0m'


def magenta(text: str) -> str:
    return '\033[35m' + text + '\033[0m'


def cyan(text: str) -> str:
    return '\033[36m' + text + '\033[0m'


def gray(text: str) -> str:
    return '\033[90m' + text + '\033[0m'
