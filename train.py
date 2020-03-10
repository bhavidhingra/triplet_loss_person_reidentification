#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import logging.config
import sys

import numpy as np
import tensorflow as tf
import math

import common

#import ipdb

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


parser.add_argument(
    '--batch_p', default=32, type=common.positive_int,
    help='The number P used in the PK-batches')

parser.add_argument(
    '--batch_k', default=4, type=common.positive_int,
    help='The number K used in PK-batches')

def show_all_parameters(log, args):
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))


def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    #ipdb.set_trace()
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    # The following simply used a subset of K of the possible FIDs
    # if >= K are available. Otherwise, we first create a padded list
    # of indices which contain a multiple of the original FID count such
    # that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
    full_range = tf.mod(tf.range(padded_count), count)

    shuffled = tf.random_shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])
    return selected_fids, tf.fill([batch_k], pid)


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

    pids, fids = common.load_dataset(args.train_set, args.image_root)

    unique_pids = np.unique(pids)
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))

    # Take the dataset size equal to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset = dataset.take((len(unique_pids) // args.batch_p) * args.batch_p)
    dataset = dataset.repeat(None)    # Repeat indefinitely.

    # For every PID, get K images.
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=args.batch_k))

    # Ungroup/flatten the batches
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    dataset = dataset.map(lambda fid, pid: common.fid_to_image(
                          fid, pid, image_root=args.image_root,
                          image_size=net_input_size),
                          num_parallel_calls=args.loading_threads)

    # Group the data into PK batches.
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    # Later elements are stored in memory while current element is being processed
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator
    images, fids, pids = dataset.make_one_shot_iterator().get_next()





if __name__ == '__main__':
    main()
