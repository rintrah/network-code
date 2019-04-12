#!/usr/bin/env python

import argparse
import keras.models as models
import pandas as pd
import numpy as np
from PIL import Image
import os.path
from tqdm import tqdm
import glob
import multiprocessing
import traceback

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('data_directory')

    p.add_argument('model')
    p.add_argument('--weights')

    p.add_argument('patch_size', type=int)
    p.add_argument('batch_size', type=int)

    p.add_argument('predictions')
    
    args = p.parse_args()

    if args.model.endswith('hd5'):
        custom_objects = dict(inspect.getmembers(losses, inspect.isfunction))
        model = models.load_model(args.model, custom_objects=custom_objects)
    elif args.model.endswith('yaml'):
        with open(args.model) as f:
            model = models.model_from_yaml(f.read())
        model.load_weights(args.weights)
    elif args.model.endswith('json'):
        with open(args.model) as f:
            model = models.model_from_json(f.read())
        model.load_weights(args.weights)
    model.summary()

    # Open dots 
    dots = pd.read_csv(os.path.join(args.data_directory, 'dots.csv'))
    dots['label'] = dots['label'].astype('category')
    labels = dots['label'].cat.categories
    
    img_counts = {label: [] for label in labels}
    img_filenames = glob.glob(os.path.join(args.data_directory, \
            'Test', '*.jpg'))
    img_ids = [int(os.path.splitext(os.path.basename(img_filename))[0])
            for img_filename in img_filenames]
    img_ids = sorted(img_ids)
    
    q = multiprocessing.Queue(maxsize=16)
    def inqueue():
        for img_id in img_ids:
            try:
                img_filename = os.path.join(args.data_directory, \
                    'Test', '%d.jpg' % img_id)

                # Open image
                img = Image.open(img_filename)
                img = img.convert('RGB')
                img = np.asarray(img, 'uint8')
                
                # Pad image so that sliding windows will match
                pad_w = args.patch_size - img.shape[1] % args.patch_size
                pad_h = args.patch_size - img.shape[0] % args.patch_size
                img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), \
                                   (pad_w // 2, pad_w - pad_w // 2), \
                                   (0, 0)), 'constant')
                
            except:
                traceback.print_exc()
                img = np.zeros((0, 0, 3), 'uint8')

            img = np.float32(img) / 127.5 - 1
            q.put(img)
        q.put(None)

    process = multiprocessing.Process(target=inqueue)
    process.start()
    
    pbar = tqdm(total=len(img_ids))
    while True:
        img = q.get()
        if img is None:
            break
        pbar.update(1)
        
        try:
            # Compute the density and counts by sliding windows
            counts = []
            densities = []                                                                     
            for y in range(0, img.shape[0], args.patch_size):
                img_row = np.array( \
                        [img[y : y + args.patch_size, x : x + args.patch_size] \
                         for x in range(0, img.shape[1], args.patch_size)])
                d, c = model.predict(img_row, args.batch_size)
                densities.append(np.hstack(d))
                counts.append(np.sum(c, axis=0))
            density = np.vstack(densities)
            count = np.sum(counts, axis=0)
            
        except:
            traceback.print_exc()
            count = np.array([0 for _ in labels])

        # Save counts
        for label, label_count in zip(labels, count):
            label_count = max(int(round(label_count)), 0)
            img_counts[label].append(label_count)

    pbar.close()
    process.join()
    
    # Replace column by plural name
    old_columns = ['adult_male', 'subadult_male', \
            'adult_female', 'juvenile', 'pup']
    new_columns = ['adult_males', 'subadult_males', \
            'adult_females', 'juveniles', 'pups']
    for old_column, new_column in zip(old_columns, new_columns):
        img_counts[new_column] = img_counts[old_column]
        del img_counts[old_column]
    
    # Save predictions
    predictions = pd.DataFrame( \
            {**{'test_id': img_ids}, **img_counts})
    col_order = ['test_id', 'adult_males', 'subadult_males', \
            'adult_females', 'juveniles', 'pups']
    predictions = predictions[col_order]
    predictions.to_csv(args.predictions, index=False)

