#! /usr/bin/env python

# stdlib
import argparse
import itertools
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
    parser.add_argument('--tag', default='return', help=' ')
    parser.add_argument('--use-cache', action='store_true', help=' ')
    parser.add_argument('--quiet', action='store_true', help=' ')
    parser.add_argument('--until-time', type=int, help=' ')
    parser.add_argument('--until-step', type=int, help=' ')
    main(**vars(parser.parse_args()))


def main(
        base_dir: Path,
        dirs: Path,
        tag: str,
        smoothing: int,
        use_cache: bool,
        quiet: bool,
        until_time: int,
        until_step: int,
):
    def get_event_files():
        for dir in dirs:
            yield from Path(base_dir, dir).glob('**/events*')

    def get_values(path):
        start_time = None
        iterator = tf.train.summary_iterator(str(path))
        while True:
            try:
                event = next(iterator)
                if start_time is None:
                    start_time = event.wall_time
                if until_time is not None and \
                        event.wall_time - start_time > until_time:
                    return
                if until_step is not None and event.step > until_step:
                    return
                for value in event.summary.value:
                    if value.tag == tag:
                        yield value.simple_value
            except DataLossError:
                pass
            except StopIteration:
                return

    def get_iterators(path):
        length = sum(1 for _ in get_values(path))
        n_iterators = max(1, length - smoothing)
        iterators = itertools.tee(get_values(path), n_iterators)
        for i, iterator in enumerate(iterators):
            for _ in range(i):
                next(iterator)  # each iterator has a different successive start point
                yield itertools.islice(iterator, smoothing)

    def get_averages():
        for path in get_event_files():
            for iterator in get_iterators(path):
                iterator, copy = itertools.tee(iterator)
                total = sum(1 for _ in copy)
                if total > 0:
                    yield path, sum(iterator) / total
                else:
                    yield None

    print('Sorted lowest to highest:')
    print('*************************')
    for path, data in sorted(get_averages()):
        if data is not None:
            cache_path = Path(path.parent, f'{smoothing}.{tag}')
            if not use_cache or not cache_path.exists():
                if not quiet:
                    print(f'Writing {cache_path}...')
                with cache_path.open('w') as f:
                    f.write(str(data))

        if not quiet:
            if data is None:
                print('No data found in', path)
            else:
                print('{:10}: {}'.format(data, path))


if __name__ == '__main__':
    cli()
