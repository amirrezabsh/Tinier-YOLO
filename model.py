import tensorflow as tf
import numpy as np
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.models as models

class TinierYolo(tf.Module):
    def __init__(self, num_classes=80, name='tinier_yolo'):
        super().__init__(name)
        self.num_classes = num_classes
        self.model = self.build_model()

    
    def summary(self):
        self.model.summary()
        
    def fire_module(self, x, squeeze=16, expand=64):
        # Squeeze part
        x = layers.Conv2D(squeeze, (1, 1), padding='same', activation='relu')(x)
        
        # Expand part
        x1 = layers.Conv2D(expand, (1, 1), padding='same', activation='relu')(x)
        x2 = layers.Conv2D(expand, (3, 3), padding='same', activation='relu')(x)
        
        # Concat
        x = layers.Concatenate(axis=3)([x1, x2])

        return x

    def first_part(self, input_layer):
        conv1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        maxpool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        
        conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1) 
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        
        conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2) 
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        
        conv4 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool3) 
        maxpool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        
        conv5 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool4) 
        maxpool5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv5)
        
        return maxpool5, maxpool4
    
    def dense_fire_modules(self, x):
        fire1 = self.fire_module(x)
        
        fire2_input = layers.Concatenate(axis=3)([
            x,
            fire1
            ])
        fire2 = self.fire_module(fire2_input)
        
        fire3_input = layers.Concatenate(axis=3)([
            x,
            fire1,
            fire2
        ])
        fire3 = self.fire_module(fire3_input)
        
        fire4_input = layers.Concatenate(axis=3)([
            x,
            fire1,
            fire2,
            fire3
        ])
        fire4 = self.fire_module(fire4_input)
        
        fire5_input = layers.Concatenate(axis=3)([
            x,
            fire1,
            fire2,
            fire3,
            fire4
        ])
        fire5 = self.fire_module(fire5_input)
        
        return fire5
    
    def pass_through_layer(self, later_feature_map, earlier_feature_map):
        # Reshape the earlier feature map
        # Get the dimensions of the earlier_feature_map
        _, height, width, channels = earlier_feature_map.shape
        
        # Ensure the height and width are even
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Height and width of the feature map must be even")

        # Split earlier_feature_map into four tensors
        split_1 = earlier_feature_map[:, :height // 2, :width // 2, :]
        split_2 = earlier_feature_map[:, :height // 2, width // 2:, :]
        split_3 = earlier_feature_map[:, height // 2:, :width // 2, :]
        split_4 = earlier_feature_map[:, height // 2:, width // 2:, :]

        # Concatenate these tensors along the channel axis
        adjusted_earlier_feature_map = layers.Concatenate(axis=3)([split_1, split_2, split_3, split_4])

        
        # Concatenate later feature map with earlier feature map
        x = layers.Concatenate(axis=3)([later_feature_map, adjusted_earlier_feature_map])
        return x
    
    def middle_part(self, maxpool5, maxpool4):
        
        # Fire module with dense connection section
        dense_fire = self.dense_fire_modules(maxpool5)
        
        # Pass-Through layer
        pass_through = self.pass_through_layer(dense_fire, maxpool4)
        
        # Last fire module
        output = self.fire_module(pass_through)
        
        return output
    
    def later_part(self, x):
        
        # 1x1 convolution
        conv1 = layers.Conv2D(16, (1, 1), activation='relu')(x)
        
        # Upsample
        upsampled = layers.UpSampling2D()(conv1)
        
        return upsampled
        
    
    def build_model(self, input_shape=(416, 416, 3)):
        input_layer = layers.Input(shape=input_shape)        
        
        # First part
        maxpool5, maxpool4 = self.first_part(input_layer)
        
        # Middle part
        middle_output = self.middle_part(maxpool5, maxpool4)
        
        # Later part
        x = self.later_part(middle_output)
        
        # Detection layers specific to Tiny YOLOv3
        # Last convolutional layer
        x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
        # x = layers.()(x)
        
        x = layers.Conv2D(255, (1, 1), activation='linear')(x)
        
        # # Reshape output to [batch_size, grid_h, grid_w, num_anchors, num_classes + 5]
        # num_anchors = 3
        # x = layers.Reshape((13, 13, num_anchors, self.num_classes + 5))(x)

        model = models.Model(inputs=input_layer, outputs=x)
        return model