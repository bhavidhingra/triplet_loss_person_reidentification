#!/usr/bin/env python3

from argparse import ArgumentParser

import os
import sys
import time

import numpy as np
import tensorflow as tf

import common
import loss
from models import Trinet

parser = ArgumentParser()

# Required arguments
parser.add_argument('--experiment_root', default="./marketroot")
parser.add_argument('--train_set',default="data/market1501_train.csv")
parser.add_argument('--image_root', type=common.readable_directory, default="../Market-1501-v15.09.15")

# Optional with defaults.
parser.add_argument('--resume', default=False)
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_p', default=32)
parser.add_argument('--batch_k', default=4)
parser.add_argument('--net_input_height', default=256)
parser.add_argument('--net_input_width', default=128)
parser.add_argument('--learning_rate', default=3e-4, type=common.positive_float)
parser.add_argument('--train_iterations', default=25000, type=common.positive_int)
parser.add_argument('--decay_start_iteration', default=15000, type=int)
parser.add_argument('--checkpoint_frequency', default=1000, type=common.nonnegative_int)
parser.add_argument('--loading_threads', default=8)
parser.add_argument('--margin', default='soft', type=common.float_or_string)
parser.add_argument('--flip_augment', default=False)
parser.add_argument('--crop_augment', default=False)
parser.add_argument('--pre_crop_height', default=288)
parser.add_argument('--pre_crop_width', default=144)
parser.add_argument('--decay_rate', default=0.01)

def show_all_parameters( args):
    print('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        print('{}: {}'.format(key, value))


def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.math.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
    full_range = tf.math.mod(tf.range(padded_count), count)

    shuffled = tf.random.shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])
    return selected_fids, tf.fill([batch_k], pid)



def main():
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

    # # To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)
    args = parser.parse_args(args=[])
    
    show_all_parameters( args)

    if not args.train_set:
        parser.print_help()
        print("You didn't specify the 'train_set' argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        print("You didn't specify the 'image_root' argument!")
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
    dataset = dataset.unbatch()

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)
    dataset = dataset.map(lambda fid, pid: common.fid_to_image(
                          fid, pid, image_root=args.image_root,
                          image_size=pre_crop_size if args.crop_augment else net_input_size)
                          )
    
    if args.flip_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_crop(im, net_input_size + (3,)), fid, pid))


    # Group the data into PK batches.
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)
    dataiter = iter(dataset)

    model = Trinet(args.embedding_dim)
<<<<<<< HEAD
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, args.train_iterations - args.decay_start_iteration, 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
=======
    lr = args.learning_rate
    decay_rate=0.01
    def expdecay_lr():
        global lr
        lr = args.learning_rate*pow(decay_rate,max(0,epoch-args.decay_start_iteration)/float(args.train_iterations-args.decay_start_iteration))
        return lr
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,args.train_iterations - args.decay_start_iteration, 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=expdecay_lr,beta_1=0.9,beta_2=0.999,epsilon=0.0001)
>>>>>>> 621de808073f623b3eb2c5c2af840696edeb9a97
    
    writer = tf.summary.create_file_writer(args.experiment_root)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.experiment_root, max_to_keep=10)

    if args.resume:
        ckpt.restore(manager.latest_checkpoint)

    for epoch in range(args.train_iterations):
        
        images, fids, pids = next(dataiter)
        with tf.GradientTape() as tape:
            emb = model(images)
            dists = loss.cdist(emb, emb)
            losses, top1, prec, topksame, negdist, posdist = loss.batch_hard(dists, pids, args.margin, args.batch_k)
            lossavg = tf.reduce_mean(losses)
            lossnp = losses.numpy()

        with writer.as_default():
            tf.summary.scalar("loss",lossavg,step=epoch)
            tf.summary.scalar('batch_top1', top1,step=epoch)
            tf.summary.scalar('batch_prec_at_{}'.format(args.batch_k-1), prec,step=epoch)
            tf.summary.scalar('learning_rate',optimizer.lr,step=epoch)
            tf.summary.histogram('losses',losses,step=epoch)
            tf.summary.histogram('embedding_dists', dists,step=epoch)
            tf.summary.histogram('embedding_pos_dists', negdist,step=epoch)
            tf.summary.histogram('embedding_neg_dists', posdist,step=epoch)

        print('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
<<<<<<< HEAD
                ' batch-p@{}: {:.2%}'.format(
                    epoch,
                    float(np.min(lossnp)),
                    float(np.mean(lossnp)),
                    float(np.max(lossnp)),
                    args.batch_k-1, float(prec)))
        grad = tape.gradient(lossavg, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
=======
            ' batch-p@{}: {:.2%} , lr:{:.4f}'.format(
                epoch,
                float(np.min(lossnp)),
                float(np.mean(lossnp)),
                float(np.max(lossnp)),
                args.batch_k-1, float(prec),optimizer.lr.numpy()))
        

        grad = tape.gradient(lossavg,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
>>>>>>> 621de808073f623b3eb2c5c2af840696edeb9a97
        ckpt.step.assign_add(1)
        if epoch % args.checkpoint_frequency == 0:
            manager.save()
                
                


   



if __name__ == '__main__':
    main()
