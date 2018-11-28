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
    return CocoClasses('../datasets', 'annotations/instances_val2017.json')


catmap = get_cat_map()
errf = open('errors.txt','w')
errf.write(str(errors))
errf.close()


def get_files_by_category(category_id):
    return catmap.get_files_for_category(category_id)
def get_categories():
    return catmap.get_all_categories()

batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
em = embedimage.VGG()
em = em.to(device)

def compute_average_for_category(category_id):
    ds = CategoryDataSet(category_id)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    total_results = None
    i = 0
    num_batches = len(loader)
    for im in loader:
        im = im.to(device)
        results = em(im)
        results_cpu = results.to('cpu')
        results_cpu = results_cpu.sum(dim=0)
        results_cpu = results_cpu.detach().numpy()
        if total_results is None:
            total_results = np.zeros(results_cpu.shape)
        total_results += results_cpu
        del results_cpu
        del results
        del im
        i += 1
        print("Category: %2d, Batch number %4d/%4d" % (category_id, i, num_batches))
    total_results *= 1.0 / len(loader)
    torch.save(total_results, os.path.join(output_dir, "%02d" % category_id))


for category in get_categories():
    compute_average_for_category(category)
    print("Done",category)

