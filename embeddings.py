from argparse import ArgumentParser

import os


import json
import numpy as np
import tensorflow as tf

import common
from models import Trinet

import h5py

parser = ArgumentParser()


parser.add_argument('--experiment_root', default="./marketroot")

parser.add_argument('--dataset', default="data/market1501_query.csv")

parser.add_argument('--image_root', default="../Market-1501-v15.09.15/")

parser.add_argument('--checkpoint', default=None)
parser.add_argument('--loading_threads', default=8)

parser.add_argument('--batch_size', default=32)

parser.add_argument('--filename', default=None)

parser.add_argument('--embedding_dim', default=128)


parser.add_argument('--net_input_height', default=256)

parser.add_argument('--net_input_width', default=128)

args = parser.parse_args([])

if args.filename is None:
    basename = os.path.basename(args.dataset)
    args.filename = os.path.splitext(basename)[0] + '_embeddings.h5'
args.filename = os.path.join(args.experiment_root, args.filename)


_, data_fids = common.load_dataset(args.dataset, args.image_root)

net_input_size = (args.net_input_height, args.net_input_width)
# pre_crop_size = (args.pre_crop_height, args.pre_crop_width)


dataset = tf.data.Dataset.from_tensor_slices(data_fids)


dataset = dataset.map(
    lambda fid: common.fid_to_image(
        fid, tf.constant('dummy'), image_root=args.image_root,
        image_size=net_input_size),
    num_parallel_calls=args.loading_threads)


dataset = dataset.batch(args.batch_size)
dataset = dataset.prefetch(1)

model = Trinet(args.embedding_dim)
optimizer = tf.keras.optimizers.Adam()
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, args.experiment_root, max_to_keep=10)

ckpt.restore(manager.latest_checkpoint).expect_partial()

with h5py.File(args.filename, 'w') as f_out:
    emb_storage = np.zeros((len(data_fids) , args.embedding_dim), np.float32)
    start_idx = 0
    for images,fids,pids in dataset:
        emb = model(images,training=False)
        emb_storage[start_idx:start_idx+len(emb)]=emb
        start_idx+=args.batch_size
    emb_dataset = f_out.create_dataset('emb', data=emb_storage)
