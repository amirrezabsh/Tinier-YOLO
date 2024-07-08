import tensorflow_datasets as tfds
import tensorflow as tf
import os

class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_path = 'datasets/'
        self.dataset_name = dataset_name
        
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        if not os.path.exists(self.dataset_path + self.dataset_name):
            os.makedirs(self.dataset_path, exist_ok=True)
            
            
            dataset, info = self.download_dataset()
            
            return dataset, info
    
    def download_dataset(self):
        if 'voc' in self.dataset_name:
            dataset, info = tfds.load(self.dataset_name, split='train', with_info=True, data_dir=self.dataset_path)
            
            print(f'Load completed: {self.dataset_name}\n{info}')
            
            return dataset, info
        
    def preprocess(self, example):
        # Define preprocessing steps here
        image = tf.image.resize(example['image'], (416, 416))
        image = tf.cast(image, tf.float32) / 255.0
        bbox = example['objects']['bbox']
        label = example['objects']['label']
        return image, bbox, label

    def get_dataset(self, batch_size=32):
        dataset = self.dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset