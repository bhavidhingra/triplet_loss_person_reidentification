#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentTypeError

def check_directory(arg, access=os.W_OK, access_str="writeable"):
    if os.path.exists(arg):
        if os.access(arg, access):
            return arg
        else:
            raise ArgumentTypeError(
                'The provided string `{0}` is not a valid {1} path '
                'since {2} is an existing folder without {1} access.'
                ''.format(arg, access_str, arg))
    
    raise ArgumentTypeError('The provided string {} is not a valid {}'
                            ' path.'.format(arg, access_str))


def writeable_directory(arg):
    return check_directory(arg, os.W_OK, "writeable")

def readable_directory(arg):
    return check_directory(arg, os.R_OK, "readable")

def number_greater_x(arg, _type, x):
    try:
        value = _type(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, _type.__name__))

    if value > x:
        return value
    else:
        raise ArgumentTypeError ('Found {} where an {} greater than {} was required'.format(
            arg, _type.__name__, x))

def positive_int(arg):
    return number_greater_x(arg, int, 0)

def nonnegative_int(arg):
    return number_greater_x(arg, int, -1)

