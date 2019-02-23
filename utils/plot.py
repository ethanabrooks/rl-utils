#! /usr/bin/env python

# stdlib
import argparse
from pathlib import Path
from typing import Optional, List

# third party
import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='*', type=Path)
    parser.add_argument('--paths', nargs='*', type=Path)
    parser.add_argument('--base-dir', default='.runs/logdir', type=Path)
    parser.add_argument('--x-tag', default='reward')
    parser.add_argument('--y-tag', default='time-step')
    parser.add_argument('--quiet', action='store_true')
    main(**vars(parser.parse_args()))


def main(
        names: List[str],
        paths: List[Path],
        x_tag: str,
        y_tag: str,
        quiet: bool,
):
    def get_tags():
        for name, path in zip(names, paths):
            if not path.exists():
                if not quiet:
                    print(f'{path} does not exist')

            for event_path in path.glob('**/events*'):
                iterator = tf.train.summary_iterator(str(event_path))
                for event in iterator:

                    def get_tags(tag_name):
                        for value in event.summary.value:
                            if value.tag == tag_name:
                                yield value

                    yield next(zip(get_tags(x_tag), get_tags(y_tag)), name)

    sns.lineplot(x=x_tag, y=y_tag, hue='name', data=(pd.DataFrame(get_tags())))
    plt.savefig()


if __name__ == '__main__':
    cli()
