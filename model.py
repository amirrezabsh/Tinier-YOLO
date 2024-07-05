import tensorflow as tf
import numpy as np
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.layers as models

class TinierYolo(tf.Module):
    def __init__(self, name='tinier-yolo'):
        super().__init__(name)
        
        self.model = self.build_model()
        
    def fire_module(self):
        pass

    def build_model(self, input_shape=(416, 416, 3)):
        input_layer = layers.Input(shape=input_shape)        
        
        # First part
        conv1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1))(input_layer)
        maxpool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        
        conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))(maxpool1) 
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        
        conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(maxpool2) 
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        
        conv4 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(maxpool3) 
        maxpool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        
        conv5 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(maxpool4) 
        maxpool5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv5)
        
        # Middle part
        