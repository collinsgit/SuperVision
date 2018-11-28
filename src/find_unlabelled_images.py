import torch
import torchvision
import torch.utils.data
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np
import sys

import gc

from data import embedimage
from data.coco import CocoClasses

mydir = os.path.split(__file__)[0]

root = '../datasets/coco/hr'
output_dir = os.path.abspath('../datasets/coco/avg_embeddings/')

class CategoryDataSet(torch.utils.data.Dataset):
    def __init__(self, category_id, *args, **kwargs):
        self.files = list(get_files_by_category(category_id))
        super().__init__(*args, **kwargs)
    def __getitem__(self, idx):
         return transforms.ToTensor()(Image.open(os.path.join(root,self.files[idx])).convert('RGB'))
    def __len__(self):
        return len(self.files)

errors = []

def get_cat_map():
    classes = CocoClasses('../datasets', 'annotations/instances_val2017.json')
    files = os.listdir('../datasets/coco/hr')

    cat_map = {}
    for file in files:
        try:
            cat = classes[file]
        except KeyError:
            errors.append(file)
        else:
            if cat not in cat_map:
                cat_map[cat] = {file, }
            else:
                cat_map[cat].add(file)
    return cat_map


catmap = get_cat_map()
errf = open('errors.txt','w')
errf.write(str(errors))
errf.close()