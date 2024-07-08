import tensorflow as tf
import numpy as np
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.models as models
from utils import yolo_layer, non_max_suppression, build_boxes

_LEAKY_RELU = 0.1
_ANCHORS = [[10, 14], [23,   27], [37,  58],
            [81, 82], [135, 169], [344, 319]]
_MAX_OUTPUT_SIZE = 20


def LeakyReLU(inputs):
    return tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


class TinierYolo(tf.Module):
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.5, num_classes=80, name='tinier_yolo'):
        super().__init__(name)
        self.num_classes = num_classes
        self.data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.model = self.build_model()

    def summary(self):
        self.model.summary()

    def fire_module(self, x, squeeze=16, expand=64):
        # Squeeze part
        x = layers.Conv2D(squeeze, (1, 1), padding='same',
                          activation=LeakyReLU)(x)

        # Expand part
        x1 = layers.Conv2D(expand, (1, 1), padding='same',
                           activation=LeakyReLU)(x)
        x2 = layers.Conv2D(expand, (3, 3), padding='same',
                           activation=LeakyReLU)(x)

        # Concat
        x = layers.Concatenate(axis=3)([x1, x2])

        return x

    def first_part(self, input_layer):
        conv1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation=LeakyReLU, name='first_conv')(input_layer)
        maxpool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)

        conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation=LeakyReLU)(maxpool1)
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

        conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation=LeakyReLU)(maxpool2)
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)

        conv4 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation=LeakyReLU)(maxpool3)
        maxpool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)

        conv5 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation=LeakyReLU)(maxpool4)
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
            raise ValueError(
                "Height and width of the feature map must be even")

        # Split earlier_feature_map into four tensors
        split_1 = earlier_feature_map[:, :height // 2, :width // 2, :]
        split_2 = earlier_feature_map[:, :height // 2, width // 2:, :]
        split_3 = earlier_feature_map[:, height // 2:, :width // 2, :]
        split_4 = earlier_feature_map[:, height // 2:, width // 2:, :]

        # Concatenate these tensors along the channel axis
        adjusted_earlier_feature_map = layers.Concatenate(
            axis=3)([split_1, split_2, split_3, split_4])

        # Concatenate later feature map with earlier feature map
        x = layers.Concatenate(axis=3)(
            [later_feature_map, adjusted_earlier_feature_map])

        return x

    def middle_part(self, maxpool5, maxpool4):

        # Fire module with dense connection section
        dense_fire = self.dense_fire_modules(maxpool5)

        # Pass-Through layer
        pass_through = self.pass_through_layer(dense_fire, maxpool4)

        # Last fire module
        output = self.fire_module(pass_through)

        return output

    def later_part(self, x, maxpool4):

        # 1x1 convolution
        conv1 = layers.Conv2D(128, (1, 1), padding='same',
                              activation=LeakyReLU, name='after_first_output')(x)

        # Upsample
        upsampled = layers.UpSampling2D()(conv1)

        # Concat
        fire1_input = layers.Concatenate(axis=3)([
            upsampled,
            maxpool4
        ])
        fire1 = self.fire_module(fire1_input)

        fire2_input = layers.Concatenate(axis=3)([
            fire1,
            fire1_input
        ])
        fire2 = self.fire_module(fire2_input)

        conv1_input = layers.Concatenate(axis=3)([
            fire1_input,
            fire1,
            fire2
        ])
        conv1 = layers.Conv2D(256, (1, 1), padding='same',
                              activation=LeakyReLU)(conv1_input)

        return upsampled

    def build_model(self, input_shape=(416, 416, 3)):
        input_layer = layers.Input(shape=input_shape)

        # First part
        maxpool5, maxpool4 = self.first_part(input_layer)

        # Middle part
        middle_output = self.middle_part(maxpool5, maxpool4)

        detection1 = yolo_layer(middle_output,
                           n_classes=self.num_classes,
                           anchors=_ANCHORS[3:],
                           img_size=input_shape,
                           data_format=self.data_format)

        # Later part
        x = self.later_part(middle_output, maxpool4)

        detection2 = yolo_layer(x,
                           n_classes=self.num_classes,
                           anchors=_ANCHORS[:3],
                           img_size=input_shape,
                           data_format=self.data_format)

        inputs = tf.concat([detection1, detection2], axis=1)
        inputs = build_boxes(inputs)
        output = non_max_suppression(inputs,
                                            n_classes=self.num_classes,
                                            max_output_size=_MAX_OUTPUT_SIZE,
                                            iou_threshold=self.iou_threshold,
                                            confidence_threshold=self.confidence_threshold)

        return models.Model(inputs=input_layer, outputs=output)
    
    
    
    def compile_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss={'yolo_loss': self.yolo_loss},
                           metrics=['accuracy'])


    
    def iou(self, box1, box2):
        # Calculate the (x, y) coordinates of the intersection rectangle
        xi1 = tf.maximum(box1[..., 0], box2[..., 0])
        yi1 = tf.maximum(box1[..., 1], box2[..., 1])
        xi2 = tf.minimum(box1[..., 2], box2[..., 2])
        yi2 = tf.minimum(box1[..., 3], box2[..., 3])
        inter_area = tf.maximum((xi2 - xi1), 0) * tf.maximum((yi2 - yi1), 0)

        # Calculate the union area
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def yolo_loss(self, y_true, y_pred):
        # Define the YOLO loss function here
        obj_mask = tf.cast(y_true[..., 4:5], dtype=tf.bool)
        no_obj_mask = tf.cast((1 - y_true[..., 4:5]), dtype=tf.bool)

        # Calculate the xy loss
        xy_loss = tf.reduce_sum(tf.square(y_true[..., :2] - y_pred[..., :2]) * obj_mask)

        # Calculate the wh loss
        wh_loss = tf.reduce_sum(tf.square(tf.sqrt(y_true[..., 2:4]) - tf.sqrt(y_pred[..., 2:4])) * obj_mask)

        # Calculate the object loss
        obj_loss = tf.reduce_sum(tf.square(y_true[..., 4:5] - y_pred[..., 4:5]) * obj_mask)

        # Calculate the no object loss
        no_obj_loss = tf.reduce_sum(tf.square(y_true[..., 4:5] - y_pred[..., 4:5]) * no_obj_mask)

        # Calculate the class loss
        class_loss = tf.reduce_sum(tf.square(y_true[..., 5:] - y_pred[..., 5:]) * obj_mask)

        # Calculate the IOU
        iou_scores = self.iou(y_true[..., :4], y_pred[..., :4])
        iou_loss = tf.reduce_sum((1 - iou_scores) * obj_mask)

        total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss + class_loss + iou_loss
        return total_loss

    def train(self, train_dataset, val_dataset, epochs=50):
        self.model.fit(train_dataset,
                       validation_data=val_dataset,
                       epochs=epochs,
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def predict(self, images):
        return self.model.predict(images)
