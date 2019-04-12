#!/usr/bin/env python

import argparse
import keras.models as models
import keras.layers as layers
import keras.backend as K
import keras.optimizers as optimizers
from losses import *

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--learning_rate', default=1e-4, type=float)
    args = p.parse_args()
    
    x = input_img = layers.Input(shape=(None, None, 3), name='input')

    x = layers.Conv2D(32, (7, 7), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (7, 7), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (7, 7), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (7, 7), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (7, 7), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)

    density = layers.Conv2D(5, (1, 1), padding='same', name='density')(x)
    count = layers.Lambda(lambda x: K.sum(x, axis=(1, 2)), name='count')(density)
    
    model = models.Model(input_img, [density, count])
    model.compile(loss=[density_loss, count_loss], \
            optimizer=optimizers.RMSprop(args.learning_rate), \
            loss_weights=[1, 0], \
            metrics={'density':[], 'count': ['mse']})

    model.save(args.model)

