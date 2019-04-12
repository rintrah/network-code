#!/usr/bin/env python

import argparse
import os
import random
import inspect

import pandas as pd
import numpy as np

import cv2
from PIL import Image
from density_image import density_image

from keras.callbacks import ModelCheckpoint, EarlyStopping, History, \
        TensorBoard, CSVLogger
import keras.models as models
from keras import losses

def iterate(data_directory, windows, dots, downsample, sigma, shuffle=True):
    if shuffle:
        windows = windows.sample(frac=1)
    
    patch_ids = windows.index.values
    img_ids = windows['img_id'].values
    xs = windows['x'].values
    ys = windows['y'].values

    num_labels = len(dots['label'].cat.categories)

    for patch_id, img_id, x, y in zip(patch_ids, img_ids, xs, ys):
        # Open up the image
        image_filename = os.path.join( \
                data_directory, 'patches', \
                '%d.tga' % patch_id)
        image = Image.open(image_filename)
        image = image.convert('RGB')
        image = np.asarray(image, 'uint8')
        image = np.float32(image) / 127.5 - 1
        
        # Find dots in this patch
        patch_dots = dots[dots['img_id'] == img_id]
        patch_xs = (np.float32(patch_dots['x'].values) - x) / downsample
        patch_ys = (np.float32(patch_dots['y'].values) - y) / downsample
        patch_labels = patch_dots['label'].cat.codes

        # Compute density
        density_shape = (image.shape[0] // downsample, \
                         image.shape[1] // downsample, \
                         num_labels)
        density = np.empty(density_shape, np.float32)
        for code in range(num_labels):
            mask = patch_labels == code
            pts = patch_xs[mask, None], patch_ys[mask, None]
            pts = np.concatenate(pts, axis=1)
            density[..., code] = density_image(pts, density_shape[: -1], sigma / downsample)

        # Random flips + transpose
        if random.random() > .5:
            image = np.flipud(image)
            density = np.flipud(density)
        if random.random() > .5:
            image = np.fliplr(image)
            density = np.fliplr(density)
        if random.random() > .5:
            image = np.rot90(image)
            density = np.rot90(density)

        yield image, (density, np.sum(density, axis=(0, 1)))

def batches(create_iterator, batch_size):
    while True:
        inputs_batch = None
        outputs_batch = None

        for inputs, outputs in create_iterator():
            # Append to inputs buffer
            if isinstance(inputs, tuple):
                if inputs_batch is None:
                    inputs_batch = tuple([element] for element in inputs)
                else:
                    for batch, element in zip(inputs_batch, inputs):
                        batch.append(element)
            else:
                if inputs_batch is None:
                    inputs_batch = [inputs]
                else:
                    inputs_batch.append(inputs)

            # Append to outputs buffer
            if isinstance(outputs, tuple):
                if outputs_batch is None:
                    outputs_batch = tuple([element] for element in outputs)
                else:
                    for batch, element in zip(outputs_batch, outputs):
                        batch.append(element)
            else:
                if outputs_batch is None:
                    outputs_batch = [outputs]
                else:
                    outputs_batch.append(outputs)

            if len(inputs_batch) >= batch_size:
                if isinstance(inputs_batch, tuple):
                    inputs_batch = [np.array(batch) for \
                            batch in inputs_batch]
                else:
                    inputs_batch = np.array(inputs_batch)

                if isinstance(outputs_batch, tuple):
                    outputs_batch = [np.array(batch) \
                            for batch in outputs_batch]
                else:
                    outputs_batch = np.array(outputs_batch)

                yield inputs_batch, outputs_batch
                inputs_batch = None
                outputs_batch = None

        if inputs_batch is not None:
            if isinstance(inputs_batch, tuple):
                inputs_batch = [np.array(batch) \
                        for batch in inputs_batch]
            else:
                inputs_batch = np.array(inputs_batch)

            if isinstance(outputs_batch, tuple):
                outputs_batch = [np.array(batch) \
                        for batch in outputs_batch]
            else:
                outputs_batch = np.array(outputs_batch)

            yield inputs_batch, outputs_batch
            inputs_batch = None
            outputs_batch = None

def create_generators(data_directory, downsample, sigma, batch_size):
    # Read in the dots
    dots = pd.read_csv(os.path.join(data_directory, 'dots.csv'))
    dots['label'] = dots['label'].astype('category')

    # Read in windows
    windows = pd.read_csv(os.path.join(data_directory, 'windows.csv'))
    
    # Open training/validation split
    in_train_filename = os.path.join(data_directory, 'in_train.csv')
    in_train = pd.read_csv(in_train_filename, index_col=0)
    train_img_ids = in_train.index[in_train['in_train']].values
    val_img_ids = in_train.index[~in_train['in_train']].values
    
    # Split dots using splitted image ids
    train_dots = dots[dots['img_id'].isin(train_img_ids)]
    val_dots = dots[dots['img_id'].isin(val_img_ids)]
    
    # Split windows using splitted image ids
    train_windows = windows[windows['img_id'].isin(train_img_ids)]
    val_windows = windows[windows['img_id'].isin(val_img_ids)]

    # Create data generator
    create_train_iterator = lambda : iterate( \
            data_directory, train_windows, train_dots, \
            downsample, sigma, True)
    create_val_iterator = lambda : iterate( \
            data_directory, train_windows, train_dots, \
            downsample, sigma, True)
    train_generator = batches(create_train_iterator, batch_size)
    val_generator = batches(create_val_iterator, batch_size)
    train_batches = (len(train_windows) + batch_size - 1) \
            // batch_size
    val_batches = (len(val_windows) + batch_size - 1) \
            // batch_size

    return train_generator, val_generator, train_batches, val_batches

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_directory')
    p.add_argument('model')
    p.add_argument('density_scale', type=int)
    p.add_argument('sigma', type=float)

    p.add_argument('--history')
    
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--epochs', default=64, type=int)
    
    p.add_argument('--cont', action='store_true')
    args = p.parse_args()
    
    train_generator, val_generator, train_batches, val_batches = \
            create_generators(args.data_directory, \
                              args.density_scale, args.sigma, args.batch_size)
    
    # Setup callbacks
    early_stopping = EarlyStopping(patience=5, verbose=2)
    model_checkpoint = ModelCheckpoint(args.model, save_best_only=True, verbose=2)
    callbacks = [early_stopping, model_checkpoint]
    if args.history is not None:
        csv_logger = CSVLogger(args.history, append=True)
        callbacks.append(csv_logger)
    
    # Load model
    custom_objects = dict(inspect.getmembers(losses, inspect.isfunction))
    model = models.load_model(args.model, custom_objects=custom_objects)
    model.summary()

    # Get score
    if args.cont:
        losses = model.evaluate_generator(val_generator, val_batches)
        val_loss_idx = model.metrics_names.index('loss')
        print('Loaded model with: %s' % ' - '.join( \
                'val_%s: %0.4f' % (metric_name, loss) \
                 for metric_name, loss in zip(model.metrics_names, losses)))
        
        # Update callbacks
        model_checkpoint.best = losses[val_loss_idx]
        early_stopping.best = losses[val_loss_idx]
        
    model.fit_generator(train_generator, \
            train_batches, \
            epochs=args.epochs, \
            callbacks=callbacks, \
            validation_data=val_generator, \
            validation_steps=val_batches)

