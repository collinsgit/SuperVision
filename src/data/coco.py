import os
import json
from data import srdata


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

        super().__init__(args, name=name, train=train, benchmark=benchmark)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'coco')
        self.dir_hr=os.path.join(self.apath,'hr')
        self.dir_lr=os.path.join(self.apath,'lr')
        self.ext = ('.jpg','.jpg')

    def _scan(self):
        names_hr, names_lr = super()._scan()
        return names_hr[self.begin:self.end], [names_lr[i][self.begin:self.end] for i in range(len(names_lr))]


class CocoClasses:
    def __init__(self, dir_data, cap_dir):
        path = os.path.join(dir_data, 'coco', cap_dir)
        annotations = json.load(open(path))['annotations']
        for entry in annotations:

    def __getitem__(self, filename):
        pass

