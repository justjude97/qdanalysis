"""
methods that build keras models
"""

import tensorflow as tf
import tensorflow.keras as keras

def adapt_resnet50(input_shape, num_classes=10, freeze_layers=True, weights='imagenet'):

    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights=weights,
                                                      input_shape=input_shape)
    
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False

    input = keras.layers.Input(shape=input_shape)
    base = base_model(input)
    #TODO make number of layers adjustable for hyper parameters
    pooling_layer = keras.layers.GlobalAveragePooling2D()(base)
    hidden_1 = keras.layers.Dense(1000, activation='relu')(pooling_layer)
    output = keras.layers.Dense(num_classes, activation='softmax')(hidden_1)

    return keras.Model(inputs=input, outputs=output)