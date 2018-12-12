from data import embedimage
from data import coco
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import operator
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance

class CategoryDataset(Dataset):
    def __init__(self, imageids, datafolder):
        self.files = sorted(imageids, key=lambda x:(x[1], x[0]))
        self.datafolder = datafolder
        self.tsfm = transforms.ToTensor()
        self._populate_avg_embeddeding()
    def _populate_avg_embeddeding(self):
        embdir = '../datasets/coco/%s/avg_embeddings' % self.datafolder
        allcats = {int(os.path.splitext(p)[0]): torch.load(os.path.join(embdir, p)) for p in os.listdir(embdir)}
        self.avg = allcats

    def __getitem__(self, i):
        imid, catid = self.files[i]
        lr = '../datasets/coco/%s/lr/X4/%012dx4.jpg'  % (self.datafolder, imid)
        hr = '../datasets/coco/%s/hr/%012d.jpg' % (self.datafolder, imid)
        avg = self.avg[catid]
        lrim = Image.open(lr).resize((300,300)).convert('RGB')
        hrim = Image.open(hr).convert('RGB')
        return catid, self.tsfm(hrim),self.tsfm(lrim), avg

    def __len__(self):
        return len(self.files)


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    datafolder = 'val'
    embedder = embedimage.VGG()
    embedder.to(device)
    NUM_TESTS_PER_CATEGORY = 10
    classes = coco.CocoClasses.make('../datasets/coco/annotations/instances_%s2017.json' % datafolder)
    all_to_load = []
    for each_category in classes.get_all_categories():
        # All the images for this category
        image_ids = list(classes.get_images_for_category(each_category))
        random.shuffle(image_ids)
        image_ids = image_ids[:NUM_TESTS_PER_CATEGORY]
        all_to_load.extend([(imid, each_category) for imid in image_ids])
    loader = DataLoader(CategoryDataset(all_to_load, datafolder), batch_size=16)
    avdistances = {}
    lowdistances = {}
    for catid, hr, lr, avg in tqdm(loader):
        hr = hr.to(device)
        lr = lr.to(device)
        hr_embedding = embedder(hr)
        lr_embedding = embedder(lr)
        hr_embedding = hr_embedding.detach().cpu().numpy()
        lr_embedding = lr_embedding.detach().cpu().numpy()
        avg = avg.numpy()
        for row_number in range(len(lr_embedding)):
            rcat, rl, rh, ra = catid[row_number], lr_embedding[row_number], hr_embedding[row_number], avg[row_number]
            # Each of these is 128 x 150 x 150
            avdiff = (ra - rh)
            lowdiff = (rl - rh)
            twos = 2 * np.ones(avdiff.shape)
            avdiff = np.power(avdiff, twos)
            lowdiff = np.power(lowdiff, twos)
            if rcat not in avdistances:
                avdistances[rcat] = avdiff
                lowdistances[rcat] = lowdiff
            else:
                avdistances[rcat] += avdiff
                lowdistances[rcat] += lowdiff
    
    for catid in avdistances.keys():
        avdistances[catid] /= NUM_TESTS_PER_CATEGORY
        lowdistances[catid] /= NUM_TESTS_PER_CATEGORY

    torch.save(avdistances, 'avdist.pt')
    torch.save(lowdistances, 'lodist.pt')


def do_compare():
    avdistances = torch.load('avdist.pt')
    lowdistances = torch.load('lodist.pt')
    

if __name__ == "__main__":
    main()
    #do_compare()