import torch
import torchvision
import torch.utils.data
from torchvision import datasets, transforms
from PIL import Image
import os
import sys

from data import embedimage

mydir = os.path.split(__file__)[0]

root = '../datasets/coco/deep'
output_dir = os.path.abspath('../datasets/coco/avg_embeddings/')

class CategoryDataSet(torch.utils.data.Dataset):
    def __init__(self, category_id, *args, **kwargs):
        self.files = get_files_by_category(category_id)
        super().__init__(*args, **kwargs)
    def __getitem__(self, idx):
         return transforms.ToTensor()(Image.open(self.files[idx]))
    def __len__(self):
        return len(self.files)

def get_files_by_category(category_id):
    return [
        '..\\datasets\\coco\\hr\\000000000139.jpg',
        '..\\datasets\\coco\\hr\\000000000285.jpg',
        '..\\datasets\\coco\\hr\\000000000632.jpg'
            ]

batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
em = embedimage.VGG()
em = em.to(device)

def compute_average_for_category(category_id):
    ds = CategoryDataSet(category_id)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    total_results = None
    for im in loader:
        im = im.to(device)
        results = em(im)
        results_cpu = results.to('cpu')
        results_cpu = results_cpu.sum(dim=0)
        if total_results is None:
            total_results = results_cpu
        else:
            total_results += results_cpu
    total_results *= 1.0 / len(loader)
    torch.save(total_results, os.path.join(output_dir, "%02d" % category_id))




'''
os.makedirs(output_dir,exist_ok=True)
base_transform = transforms.Compose([
    transforms.ToTensor()
    ])
images = datasets.ImageFolder(root=os.path.join(mydir,root), transform=base_transform)

batch_size=32

loader = torch.utils.data.DataLoader(images, batch_size=batch_size)

em = embedimage.VGG()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
em = em.to(device)
filenames = [imname for imname, _ in images.imgs]

i = 0

for (batch_num, (im, _)) in enumerate(loader):
    im = im.to(device)
    results = em(im)
    results = results.to('cpu')
    for j in range(len(results)):
        outpath = os.path.join(output_dir, '%s.pt' % os.path.splitext(os.path.basename(filenames[i]))[0])
        torch.save(results[j], outpath)
        i += 1
    print("Done %d batches" % batch_num)


'''