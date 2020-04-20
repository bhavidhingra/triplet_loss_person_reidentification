from __future__ import print_function
from argparse import ArgumentParser
import os

import h5py
import json
import numpy as np
import tensorflow as tf

import common
import loss
import matplotlib.pyplot as plt


parser = ArgumentParser()

parser.add_argument('--query_dataset',default='data/market1501_query.csv')

parser.add_argument('--query_embeddings',default='/home/mythri/Downloads/marketroot2/market1501_query_embeddings.h5')

parser.add_argument('--gallery_dataset',default='data/market1501_test.csv')

parser.add_argument('--gallery_embeddings',default='/home/mythri/Downloads/marketroot2/market1501_test_embeddings.h5')

parser.add_argument('--filename',default='/home/mythri/Downloads/marketroot2/evalresults.json')

parser.add_argument('--image_root',default='../Market-1501-v15.09.15')

parser.add_argument('--batch_size', default=256)



def average_precision(y_true, y_score):
    y_true_sorted = y_true[np.argsort(-y_score, kind='mergesort')]

    tp = np.cumsum(y_true_sorted)
    total_true = np.sum(y_true_sorted)
    recall = tp / float(total_true)
    recall = np.insert(recall, 0, 0.)
    precision = tp / np.arange(1, len(tp) + 1,dtype=np.float)
    precision = np.insert(precision, 0, 1.)
    ap = np.sum(np.diff(recall) * ((precision[1:] + precision[:-1]) / 2))

    return ap


def main():
    
    args = parser.parse_args([])

    
    query_pids, query_fids = common.load_dataset(args.query_dataset, None)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, None)

    # Load the two datasets fully into memory.
    with h5py.File(args.query_embeddings, 'r') as f_query:
        query_embs = np.array(f_query['emb'])
    with h5py.File(args.gallery_embeddings, 'r') as f_gallery:
        gallery_embs = np.array(f_gallery['emb'])


    dataset = tf.data.Dataset.from_tensor_slices((query_pids, query_fids, query_embs))
    dataset=dataset.batch(args.batch_size)


    aps = []
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    fnames={}
    uniqpids = []
    start_idx=0
    for pids,fids,embs in dataset:
        try:
            distances = loss.cdist(embs,gallery_embs)
            
            print('\rEvaluating batch {}-{}/{}'.format(
                    start_idx, start_idx + len(fids), len(query_fids)))
            start_idx+=len(fids)
        except tf.errors.OutOfRangeError:
            print()  
            break

        pids, fids = np.array(pids, '|U'), np.array(fids, '|U')
        
        pid_matches = gallery_pids[None] == pids[:,None]


        scores = 1.0 / (1 + distances)
        for i in range(len(distances)):
            ap = average_precision(pid_matches[i], scores[i])

            if np.isnan(ap):
                continue

            aps.append(ap)
            sorteddist = np.argsort(distances[i])
            k = np.where(pid_matches[i,sorteddist])[0][0]
            if len(fnames)<5:
                if pids[i] not in uniqpids:
                    uniqpids.append(pids[i])
                    temp=np.where(pid_matches[i,sorteddist])[0]
                    if len(temp)>=3: 
                        fnames[fids[i]]=list(np.array(gallery_fids[sorteddist][temp[:3]],'|U'))
            cmc[k:] += 1

    cmc = cmc / float(len(query_pids))
    mean_ap = np.mean(aps)

    
    
    print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
        mean_ap, cmc[0], cmc[1], cmc[4], cmc[9]))

    plt.figure(figsize=(20,15))
    i=1
    for key,val in fnames.items():
        plt.subplot(5,4,4*(i-1)+1)
        img =plt.imread(os.path.join(args.image_root,key))
        plt.imshow(img)
        # print(key)
        # print(img == None)
        if i == 1:
            plt.title('query')
        plt.axis('off')
        for j in range(3):
            plt.subplot(5,4,(i-1)*4+j+2)
            plt.imshow(plt.imread(os.path.join(args.image_root,val[j])))
            if i == 1 and j == 1:
                plt.title('matches')
            plt.axis('off')
        i+=1
    plt.show()

    if args.filename is not None:
        with open(args.filename,'w') as f:
            json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)}, f)
   

if __name__ == '__main__':
    main()