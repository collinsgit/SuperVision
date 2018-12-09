import os
import json
from data import srdata
import torch
import numpy as np
import sys
import random
from tqdm import tqdm


class Coco(srdata.SRData):
    def __init__(self, args, name='coco',train=True,benchmark=False):
        self.train = train
        self.args = args

        self._set_filesystem(args.dir_data)
        self.classes = CocoClasses.make(os.path.join(
                self.apath,
                '..',
                'annotations',
                'instances_%s2017.json' % ('train' if self.train else 'val')
            ),
            self.dir_hr
        )
        super().__init__(args, name=name, train=train, benchmark=benchmark)
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

    def _get_datarange(self):
        def get_filenames_from_imid(imid):
            return (os.path.join(self.dir_hr,'%012d%s' % (imid,self.ext[0])),
                    [
                        os.path.join(self.dir_lr, 'X%d' % (s),'%012dx%d%s' % (imid, s, self.ext[1]))  
                        for s in self.args.scale
                    ])

        all_categories = sorted(list(self.classes.get_all_categories()))
        if self.args.randomize_categories:
            random.shuffle(all_categories)
        data_range = self.args.data_range.split('/')
        assert len(data_range) > 0, "Data range must not be empty"
        if self.train:
            data_range = data_range[0]
        else:
            data_range = data_range[-1]
        images_to_load = set()
        categories_to_load = {}
        if data_range == 'all':
            images_to_load = self.classes.get_all_image_ids()
        else:
            for component in data_range.split('+'):
                category_spec, num_per_category = component.split(',')
                num_per_category = int(num_per_category)
                # Range, not random
                for k in range(*map(int,category_spec.split('-'))):
                    k = all_categories[k]
                    if k not in categories_to_load:
                        categories_to_load[k] = 0
                    categories_to_load[k] += num_per_category
        
        for each_cat in categories_to_load.keys():
            images = list(sorted(self.classes.get_images_for_category(each_cat)))
            if self.args.randomize_category_picks:
                random.shuffle(images)
            images_to_load.update(set(images[:categories_to_load[each_cat]]))
        print("Loading",images_to_load)
        return list(map(get_filenames_from_imid,images_to_load))

        


    def _scan(self):
        dr = self._get_datarange()
        print(dr)
        names_hr, all_lr = list(zip(*self._get_datarange()))
        names_lr = list(zip(*all_lr))
        print(names_hr, names_lr)
        return names_hr, names_lr

    def __getitem__(self, idx):
        lr, hr, fname, _ = super().__getitem__(idx)
        avg_classification = None
        for c in sorted(list(self.classes[fname])):
            avg_classification = self.avg_by_class[c]
            break
        # avg_classification = np.average([self.avg_by_class[c] for c in self.classes[fname]], axis=0)
        return lr, hr, fname, avg_classification


class CocoClasses(object):
    def __init__(self, captionpath, hrpath = None):
        path = captionpath
        annotations = json.load(open(path))['annotations']

        self.categories_by_file = {}
        self.files_by_category = {}
        print("Parsing annotation file")
        for entry in tqdm(annotations):
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
        yield from list(self.files_by_category.keys())

    def filename_for_image_id(self, image_id):
        return "%012d.jpg" % image_id

    def filterby(self, hrpath):
        print("Filtering images...")
        for imid in tqdm(list(self.get_all_image_ids())):
            if not os.path.isfile(os.path.join(hrpath,self.filename_for_image_id(imid))):
                # We don't actually have this image
                categories_to_modify = list(self.get_categories_for_image_id(imid))
                del self.categories_by_file[imid]
                for cat in categories_to_modify:
                    self.files_by_category[cat].remove(imid)
        return self

    @staticmethod 
    def make(captionpath, hrpath):
        saved = '%s.pt' % os.path.splitext(captionpath)[0]
        if os.path.isfile(saved):
            print("Using cached annotation file for:\n%s" % captionpath)
            obj = torch.load(saved)
            return obj
        else:
            obj = CocoClasses(captionpath)
            obj.filterby(hrpath)
            torch.save(obj, saved)
            return obj

