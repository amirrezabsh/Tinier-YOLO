import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

_LEAKY_RELU = 0.1
_ANCHORS = [[10, 14], [23,   27], [37,  58],
            [81, 82], [135, 169], [344, 319]]
_MAX_OUTPUT_SIZE = 20


def LeakyReLU(inputs):
    return tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
YOLO_STRIDES                = [8, 16, 32]

YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                            [[30,  61], [62,   45], [59,  119]],
                            [[116, 90], [156, 198], [373, 326]]]
STRIDES         = np.array(YOLO_STRIDES)
ANCHORS         = (np.array(YOLO_ANCHORS).T/STRIDES).T
def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales  
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

class TinierYolo(tf.Module):
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.5, num_classes=80, name='tinier_yolo'):
        super().__init__(name)
        self.num_classes = num_classes
        self.data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.training = True
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

        # detection1 = yolo_layer(middle_output,
        #                    n_classes=self.num_classes,
        #                    anchors=_ANCHORS[3:],
        #                    img_size=input_shape,
        #                    data_format=self.data_format)

        # Later part
        x = self.later_part(middle_output, maxpool4)

        # detection2 = yolo_layer(x,
        #                    n_classes=self.num_classes,
        #                    anchors=_ANCHORS[:3],
        #                    img_size=input_shape,
        #                    data_format=self.data_format)

        # inputs = tf.concat([detection1, detection2], axis=1)
        # inputs = build_boxes(inputs)
        
        # boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        #     boxes=tf.reshape(inputs[:, :, :4], (tf.shape(inputs)[0], -1, 1, 4)),
        #     scores=tf.reshape(inputs[:, :, 4:], (tf.shape(inputs)[0], -1, tf.shape(inputs)[-1] - 4)),
        #     max_output_size_per_class=_MAX_OUTPUT_SIZE,
        #     max_total_size=_MAX_OUTPUT_SIZE,
        #     iou_threshold=self.iou_threshold,
        #     score_threshold=self.confidence_threshold
        # )
        conv_tensors = [0, 1]
        
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            if i == 0:
              conv_tensor = x
            else:
              conv_tensor = middle_output
            pred_tensor = decode(conv_tensor, self.num_classes, i)
            # if self.training: output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)
                
        return models.Model(inputs=input_layer, outputs=output_tensors)
    
    
    
    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss=self.yolo_loss,
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
        lambda_coord = 5.0
        lambda_noobj = 0.5

        print(y_true.shape)
        print(y_pred.shape)
        # Extract ground truth values
        true_boxes = y_true[..., 0:4]
        true_confidence = y_true[..., 4]
        true_classes = y_true[..., 5:]

        # Extract predicted values
        pred_boxes = y_pred[..., 0:4]
        pred_confidence = y_pred[..., 4]
        pred_classes = y_pred[..., 5:]

        # Compute IoU between true and predicted boxes
        iou = self.iou(true_boxes, pred_boxes)

        # Localization loss (sum of squared errors)
        coord_loss = lambda_coord * tf.reduce_sum(tf.square(true_boxes - pred_boxes), axis=-1)

        # Confidence loss (sum of squared errors)
        confidence_loss = tf.reduce_sum(tf.square(true_confidence - iou), axis=-1)

        # Class probability loss (sum of squared errors)
        class_loss = tf.reduce_sum(tf.square(true_classes - pred_classes), axis=-1)

        # Total loss
        total_loss = coord_loss + confidence_loss + class_loss
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
