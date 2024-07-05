import tensorflow as tf
import numpy as np
import tf.contrib.keras.layers as layers
import tf.contrib.keras.models as models


class TinierYolo(tf.Module):
    def __init__(self, name='tinier-yolo'):
        super().__init__(name)
        
        self.model = self.build_model()
        
    def fire_module(self):
        pass

    def build_model(self, input_shape=(416, 416, 3)):
        input_layer = layers.Input(shape=input_shape)
        
        
        