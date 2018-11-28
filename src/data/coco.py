import os
import json
from data import srdata
import torch
import numpy as np

CATEGORIES_FILE = 'annotations/instances_val2017.json'


class Coco(srdata.SRData):
    def __init__(self, args, name='coco',train=True,benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))

        self.classes = CocoClasses(args.dir_data, CATEGORIES_FILE)
        self.avg_by_class = {}

        super().__init__(args, name=name, train=train, benchmark=benchmark)
        self._populate_avg_embedding()

    def _populate_avg_embedding(self):
        for each_category in self.classes.get_all_categories():
            avgp = os.path.join(self.dir_avg_embedding, '%02d.pt' % each_category)
            self.avg_by_class[each_category] = torch.load(avgp)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'coco')
        self.dir_avg_embedding = os.path.join(self.apath, 'avg_embeddings')
        self.dir_hr=os.path.join(self.apath,'hr')
        self.dir_lr=os.path.join(self.apath,'lr')
        self.ext = ('.jpg','.jpg')

    def _scan(self):
        names_hr, names_lr = super()._scan()
        return names_hr[self.begin:self.end], [names_lr[i][self.begin:self.end] for i in range(len(names_lr))]

    def __getitem__(self, idx):
        lr, hr, fname, _ = super().__getitem__(idx)
        avg_classification = np.average([self.avg_by_class[c] for c in self.classes[fname]], axis=0)
        return lr, hr, fname, avg_classification


class CocoClasses(object):
    def __init__(self, dir_data, cap_dir):
        path = os.path.join(dir_data, 'coco', cap_dir)
        annotations = json.load(open(path))['annotations']

        self.categories_by_file = {}
        self.files_by_category = {}
        for entry in annotations:
            catid = entry['category_id']
            imid = entry['image_id']
            if imid not in self.categories_by_file:
                self.categories_by_file[imid] = set()
            self.categories_by_file[imid].add(catid)
            if catid not in self.files_by_category:
                self.files_by_category[catid] = set()
            self.files_by_category[catid].add(imid) 

    def __getitem__(self, filename):
        return self.categories_by_file[int(filename.split('.')[0].lstrip('0'))]
    def get_categories_for_filename(self, filename):
        return self.categories_by_file[int(filename.split('.')[0].lstrip('0'))]
    def get_categories_for_image_id(self, image_id):
        return self.categories_by_file[image_id]
    def get_files_for_category(self, category):
        return self.files_by_category[catid]
    def get_all_image_ids(self):
        return sorted(list(self.categories_by_file.keys()))
    def get_all_categories(self):
        return sorted(list(self.files_by_category.keys()))


