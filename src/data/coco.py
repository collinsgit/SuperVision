import os
from data import srdata

class Coco(srdata.SRData):
    def __init__(self, args, name='Coco',train=True,benchmark=False):
        super().__init__(args, name=name, train=train, benchmark=benchmark)
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr=os.path.join(self.apath,'hr')
        self.dir_lr=os.path.join(self.apath,'lr')
        self.ext = ('.jpg','.jpg')