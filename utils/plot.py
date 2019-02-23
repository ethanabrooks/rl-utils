#! /usr/bin/env python

# stdlib
import argparse
from pathlib import Path
from typing import List, Optional

# third party
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='*', type=Path)
    parser.add_argument('--paths', nargs='*', type=Path)
    parser.add_argument('--base-dir', default='.runs/logdir', type=Path)
    parser.add_argument('--tag', default='return')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--limit', type=int)
    main(**vars(parser.parse_args()))


def main(
        names: List[str],
        paths: List[Path],
        tag: str,
        base_dir: Path,
        limit: Optional[int],
        quiet: bool,
):
    def get_tags():
        for name, path in zip(names, paths):
            if not path.exists():
                if not quiet:
                    print(f'{path} does not exist')

            for event_path in Path(base_dir, path).glob('**/events*'):
                iterator = tf.train.summary_iterator(str(event_path))
                for event in iterator:
                    value = event.summary.value
                    if value:
                        if value[0].tag == tag:
                            value = value[0].simple_value
                            if limit is None or event.step < limit:
                                yield event.step, value, name

    data = pd.DataFrame(get_tags())
    sns.lineplot(x=0, y=1, hue=2, data=data)
    plt.savefig('plot')


if __name__ == '__main__':
    cli()
