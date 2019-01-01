import argparse
from itertools import filterfalse
import re
from typing import Tuple

from gym import spaces
import numpy as np
import tensorflow as tf

from utils.tensorflow import parametric_relu


def parse_groups(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    def is_optional(group):
        return group.title == 'optional arguments'

    def parse_group(group):
        # noinspection PyProtectedMember
        return {
            a.dest: getattr(args, a.dest, None)
            for a in group._group_actions
        }

    # noinspection PyUnresolvedReferences,PyProtectedMember
    groups = [
        g for g in parser._action_groups if g.title != 'positional arguments'
    ]
    optional = filter(is_optional, groups)
    not_optional = filterfalse(is_optional, groups)

    kwarg_dicts = {group.title: parse_group(group) for group in not_optional}
    kwargs = (parse_group(next(optional)))
    del kwargs['help']
    return {**kwarg_dicts, **kwargs}


def parse_double(ctx, param, string):
    if string is None:
        return
    a, b = map(float, string.split(','))
    return a, b


def make_box(*tuples: Tuple[float, float]):
    low, high = map(np.array, zip(*[(map(float, m)) for m in tuples]))
    return spaces.Box(low=low, high=high, dtype=np.float32)


def parse_space(dim: int):
    def _parse_space(arg: str):
        regex = re.compile('\((-?[\.\d]+),(-?[\.\d]+)\)')
        matches = regex.findall(arg)
        if len(matches) != dim:
            raise argparse.ArgumentTypeError(
                f'Arg {arg} must have {dim} substrings '
                f'matching pattern {regex}.')
        return make_box(*matches)

    return _parse_space


def parse_vector(length: int, delim: str):
    def _parse_vector(arg: str):
        vector = tuple(map(float, arg.split(delim)))
        if len(vector) != length:
            raise argparse.ArgumentError(
                f'Arg {arg} must include {length} float values'
                f'delimited by "{delim}".')
        return vector

    return _parse_vector


def cast_to_int(arg: str):
    return int(float(arg))


ACTIVATIONS = dict(
    relu=tf.nn.relu,
    leaky=tf.nn.leaky_relu,
    elu=tf.nn.elu,
    selu=tf.nn.selu,
    prelu=parametric_relu,
    sigmoid=tf.sigmoid,
    tanh=tf.tanh,
    none=None,
)


def parse_activation(arg: str):
    return ACTIVATIONS[arg]
