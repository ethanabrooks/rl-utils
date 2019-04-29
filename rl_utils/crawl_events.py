#! /usr/bin/env python

# stdlib
import argparse
from collections import namedtuple
from itertools import islice
from pathlib import Path
from typing import List, Optional

# third party
import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dirs', nargs='*', type=Path)
    parser.add_argument(
        '--base-dir', default='.runs/logdir', type=Path, help=' ')
    parser.add_argument('--smoothing', type=int, default=2000, help=' ')
    parser.add_argument('--tag', default='reward', help=' ')
    parser.add_argument('--use-cache', action='store_true', help=' ')
    parser.add_argument('--quiet', action='store_true', help=' ')
    main(**vars(parser.parse_args()))


def main(
        base_dir: Path,
        dirs: Path,
        tag: str,
        smoothing: int,
        use_cache: bool,
        quiet: bool,
):
    data_points = crawl(
        dirs=[Path(base_dir, d) for d in dirs],
        tag=tag,
        smoothing=smoothing,
        use_cache=use_cache,
        quiet=quiet)
    print('Sorted lowest to highest:')
    print('*************************')
    for data, event_file in sorted(data_points):
        print('{:10}: {}'.format(data, event_file))


DataPoint = namedtuple('DataPoint', 'data source')


def crawl(dirs: List[Path], tag: str, smoothing: int, use_cache: bool,
          quiet: bool) -> List[DataPoint]:
    if not quiet:
        for directory in dirs:
            if not directory.exists():
                print(f'{directory} does not exist')
    event_files = collect_events_files(dirs)
    data_points = []
    for event_file_path in event_files:
        data = collect_data(
            tag=tag, event_file_path=event_file_path, n=smoothing)
        if data:
            cache_path = Path(event_file_path.parent, f'{smoothing}.{tag}')
            data_points.append((data, event_file_path))
            if not use_cache or not cache_path.exists():
                if not quiet:
                    print(f'Writing {cache_path}...')
                with cache_path.open('w') as f:
                    f.write(str(data))
        elif not quiet:
            print('Tag not found')
    return data_points


def collect_events_files(dirs):
    pattern = '**/events*'
    return [path for d in dirs for path in d.glob(pattern)]


def collect_data(tag: str, event_file_path: Path, n: int) -> Optional[float]:
    """
    :param event_file_path: path to events file
    :param n: number of data points to average
    :return: average of last n data-points in events file or None if events file is empty
    """
    try:
        length = sum(
            1 for _ in tf.train.summary_iterator(str(event_file_path)))
    except DataLossError:
        return None
    iterator = tf.train.summary_iterator(str(event_file_path))
    events = islice(iterator, max(length - n, 0), length)

    def get_tag(event):
        return next((v.simple_value
                     for v in event.summary.value if v.tag == tag), None)

    data = (get_tag(e) for e in events)
    data = [d for d in data if d is not None]
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        return None


if __name__ == '__main__':
    cli()
