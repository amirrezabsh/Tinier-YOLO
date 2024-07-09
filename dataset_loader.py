import tensorflow_datasets as tfds
import tensorflow as tf
import os

class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_path = 'datasets/'
        self.dataset_name = dataset_name
        
        self.dataset, self.info = self.load_dataset()
        
    def load_dataset(self):
        if not os.path.exists(self.dataset_path + self.dataset_name):
            os.makedirs(self.dataset_path, exist_ok=True)
        
        dataset, info = self.download_dataset()
        return dataset, info
    
    def download_dataset(self):
        if 'voc' in self.dataset_name:
            dataset, info = tfds.load(self.dataset_name, split=['train', 'test', 'validation'], with_info=True, data_dir=self.dataset_path)

            print(f'Load completed: {self.dataset_name}\n{info}')
            
            return dataset, info
    
    def preprocess(self, example):
        # Define preprocessing steps here
        image = tf.image.resize(example['image'], (416, 416))
        image = tf.cast(image, tf.float32) / 255.0
        bbox = example['objects']['bbox']
        label = example['objects']['label']
        return image, bbox, label

    def pad_boxes(self, boxes, labels, max_objects=100):
        boxes = boxes[:max_objects]
        labels = labels[:max_objects]
        boxes = tf.pad(boxes, [[0, max_objects - tf.shape(boxes)[0]], [0, 0]])
        labels = tf.pad(labels, [[0, max_objects - tf.shape(labels)[0]]])
        return boxes, labels

    def get_dataset(self, split, batch_size=16):
        def preprocess(data):
            image = tf.image.resize(data['image'], (416, 416))
            image = image / 255.0
            boxes, labels = self.pad_boxes(data['objects']['bbox'], data['objects']['label'])
            return image, boxes, labels

        dataset = self.dataset[split].map(preprocess)
        dataset = dataset.batch(batch_size)
        return dataset

