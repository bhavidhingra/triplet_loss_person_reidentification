#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentTypeError

import numpy as np

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

def get_logging_dict(name):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': name + '.log',
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['logfile'],
                'level': 'DEBUG',
                'propagate': True
            },
            # extra ones to shut up.
            'tensorflow': {
                'handlers': ['logfile'],
                'level': 'INFO',
            },
        }
    }


def load_dataset(csv_file, image_root, fail_on_missing=True):
    """ Loads a dataset .csv file, return PIDs and FIDs. 

        PIDs are the "person IDs", i.e. class names/labels.
        FIDs are the "file IDs", which are individual relative filenames.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    pids, fids = dataset.T

    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            missing[i] = not os.path.isfile(os.path.join(image_root, fid))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                              '{} images are missing'.format(
                              csv_file, image_root, missing_count, len(fids)))
            else:
                print ('[Warning] removing {} missing file(s) from the dataset'.format(missing_count))
                fids = fids[np.logical_not(missing)]
                pids = pids[np.logical_not(missing)]

    return pids, fids
