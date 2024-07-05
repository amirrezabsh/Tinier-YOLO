import tensorflow_datasets as tfds
import os

class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_path = 'datasets/'
        self.dataset_name = dataset_name
        
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        if not os.path.exists(self.dataset_path + self.dataset_name):
            os.makedirs(self.dataset_path, exist_ok=True)
            
            
            dataset = self.download_dataset()
            
            return dataset
    
    def download_dataset(self):
        if 'voc' in self.dataset_name:
            dataset, info = tfds.load(self.dataset_name, split='train', with_info=True, data_dir=self.dataset_path)
            
            print(f'Load completed: {self.dataset_name}\n{info}')
            
            return dataset
        