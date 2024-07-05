from dataset_loader import DatasetLoader

if __name__ == '__main__':
    pascal_voc_2007 = 'voc/2007'
    pascal_voc_2012 = 'voc/2012'
    
    loader2007 = DatasetLoader(pascal_voc_2007)
    loader2012 = DatasetLoader(pascal_voc_2012)
    
    