#! /usr/bin/env python

# stdlib
import argparse
from pathlib import Path
from typing import List, Optional

# third party
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
import seaborn as sns


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--names', nargs='*', type=Path)
    parser.add_argument('--paths', nargs='*', type=Path)
    parser.add_argument('--base-dir', default='.runs/logdir', type=Path)
    parser.add_argument('--tag', default='return')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--fname', type=str, default='plot')
    parser.add_argument('--quality', type=int)
    parser.add_argument('--dpi', type=int, default=256)
    main(**vars(parser.parse_args()))


def main(
        names: List[str],
        paths: List[Path],
        tag: str,
        base_dir: Path,
        limit: Optional[int],
        quiet: bool,
        **kwargs,
):
    def get_tags():
        for name, path in zip(names, paths):
            path = Path(base_dir, path)
            if not path.exists():
                if not quiet:
                    print(f'{path} does not exist')

            for event_path in path.glob('**/events*'):
                iterator = tf.train.summary_iterator(str(event_path))
                for event in iterator:
                    value = event.summary.value
                    if value:
                        if value[0].tag == tag:
                            value = value[0].simple_value
                            if limit is None or event.step < limit:
                                yield event.step, value, name

    data = pd.DataFrame(get_tags(), columns=['step', tag, 'run'])
    sns.lineplot(x='step', y=tag, hue='run', data=data)
    plt.legend(data['run'].unique())
    plt.axes().ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    plt.savefig(**kwargs)


if __name__ == '__main__':
    cli()
