import os
import json
from data import srdata
import torch
import numpy as np
import sys


class Coco(srdata.SRData):
    def __init__(self, args, name='coco',train=True,benchmark=False):
        if args.data_range == 'all':
            self.begin, self.end = None, None
        else:
            data_range = [r.split('-') for r in args.data_range.split('/')]
            if train:
                data_range = data_range[0]
            else:
                if args.test_only and len(data_range) == 1:
                    data_range = data_range[0]
                else:
                    data_range = data_range[1]

            self.begin, self.end = list(map(lambda x: int(x), data_range))

        super().__init__(args, name=name, train=train, benchmark=benchmark)

        self.classes = CocoClasses(os.path.join(
            self.apath,
            '..',
            'annotations',
            'instances_%s2017.json' % ('train' if self.train else 'val')),
            self.dir_hr
            )
        self.avg_by_class = {}
        self._populate_avg_embedding()

    def _populate_avg_embedding(self):
        for each_category in self.classes.get_all_categories():
            avgp = os.path.join(self.dir_avg_embedding, '%02d.pt' % each_category)
            self.avg_by_class[each_category] = torch.load(avgp)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'coco', 'train' if self.train else 'val')
        self.dir_avg_embedding = os.path.join(self.apath, 'avg_embeddings')
        self.dir_hr=os.path.join(self.apath,'hr')
        self.dir_lr=os.path.join(self.apath,'lr')
        self.ext = ('.jpg','.jpg')

    def _scan(self):
        names_hr, names_lr = super()._scan()
        if self.begin is None or self.end is None:
            return names_hr, names_lr
        return names_hr[self.begin:self.end], [names_lr[i][self.begin:self.end] for i in range(len(names_lr))]

    def __getitem__(self, idx):
        lr, hr, fname, _ = super().__getitem__(idx)
        avg_classification = np.average([self.avg_by_class[c] for c in self.classes[fname]], axis=0)
        return lr, hr, fname, avg_classification


class CocoClasses(object):
    def __init__(self, captionpath, hrpath = None):
        path = captionpath
        annotations = json.load(open(path))['annotations']

        self.categories_by_file = {}
        self.files_by_category = {}
        for entry in annotations:
            catid = entry['category_id']
            imid = entry['image_id']
            if hrpath is not None and not os.path.isfile(os.path.join(hrpath,self.filename_for_image_id(imid))):
                continue
            if imid not in self.categories_by_file:
                self.categories_by_file[imid] = set()
            self.categories_by_file[imid].add(catid)
            if catid not in self.files_by_category:
                self.files_by_category[catid] = set()
            self.files_by_category[catid].add(imid) 

    def __getitem__(self, filename):
        return self.categories_by_file[int(os.path.splitext(os.path.basename(filename))[0])]

    def get_categories_for_filename(self, filename):
        yield from self.categories_by_file[int(os.path.splitext(os.path.basename(filename))[0])]

    def get_categories_for_image_id(self, image_id):
        yield from self.categories_by_file[image_id]

    def get_images_for_category(self, category):
        yield from self.files_by_category[category]

    def get_files_for_category(self, category):
        yield from map(self.filename_for_image_id, self.get_images_for_category(category))

    def get_all_image_ids(self):
        yield from self.categories_by_file.keys()

    def get_all_filenames(self):
        yield from map(self.filename_for_image_id, self.categories_by_file.keys())

    def get_all_categories(self):
        yield from self.files_by_category.keys()

    def filename_for_image_id(self, image_id):
        return "%012d.jpg" % image_id


