import tensorflow as tf
from tensorflow import keras
import numpy as np


def graph(input_X, input_Y):
    input = keras.layers.Input((input_X, input_Y, 3))

    Conv1 = keras.layers.Conv2D(16, (3, 3), activation="relu", padding='same')(input)
    Conv1 = keras.layers.Conv2D(16, (3, 3), activation="relu", padding='same')(Conv1)
    Pool1 = keras.layers.MaxPooling2D((2, 2)) (Conv1)

    Conv2 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same')(Pool1)
    Conv2 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same')(Conv2)
    Pool2 = keras.layers.MaxPooling2D((2, 2))(Conv2)    

    Conv3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same')(Pool2)
    Conv3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same')(Conv3)
    Pool3 = keras.layers.MaxPooling2D((2, 2))(Conv3)

    Conv4 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same')(Pool3)
    Conv4 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same')(Conv4)
    Pool4 = keras.layers.MaxPooling2D((2, 2))(Conv4)

    Conv5 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding='same')(Pool4)
    Conv5 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding='same')(Conv5)
    Pool5 = keras.layers.MaxPooling2D((2, 2))(Conv5)
    
    Conv6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(Pool5)
    Conv6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(Conv6)
    
    Ups0 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2))(Conv6)
    Ups0 = keras.layers.Concatenate()([Ups0, Conv5])
    Ups0_conv = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(Ups0)
    Ups0_conv = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(Ups0_conv)
  
    Ups1 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(Ups0_conv)
    Ups1 = keras.layers.Concatenate()([Ups1, Conv4])
    Ups1_conv = keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same')(Ups1)
    Ups1_conv = keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same')(Ups1_conv)

    Ups2 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(Ups1_conv)
    Ups2 = keras.layers.Concatenate()([Ups2, Conv3])
    Ups2_conv = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(Ups2)
    Ups2_conv = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(Ups2_conv)
    
    Ups3 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(Ups2_conv)
    Ups3 = keras.layers.Concatenate()([Ups3, Conv2])
    Ups3_conv = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(Ups3)
    Ups3_conv = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(Ups3_conv)

    Ups4 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(Ups3_conv)
    Ups4 = keras.layers.Concatenate()([Ups4, Conv1])
    Ups4_conv = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(Ups4)
    Ups4_conv = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(Ups4_conv)

    output_logit = keras.layers.Conv2D(1, (1, 1))(Ups4_conv)

    return keras.Model(inputs=input, outputs=output_logit)
