#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import logging.config
import sys

import common

parser = ArgumentParser(description='Train a triplet loss person re-identification network.')

# Required arguments
parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--train_set',
    help='Path to the train_set csv file.')

parser.add_argument(
    '--image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv')

# Optional with defaults.
parser.add_argument(
    '--resume', action='store_true', default=False,
    help='With this flag, all other arguments apart from the experiment_root'
         'are ignored and a previously saved set of arguments is loaded.')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default=None,
    help='Path to the checkpoint file of the pretrained network.')


def show_all_parameters(log, args):
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))


def main():
    args = parser.parse_args()
    
    log_file = os.path.join(args.experiment_root, "train")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('train')

    show_all_parameters(log, args)

    if not args.train_set:
        parser.print_help()
        log.error("You didn't specify the 'train_set' argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        log.error("You didn't specify the 'image_root' argument!")
        sys.exit(1)
