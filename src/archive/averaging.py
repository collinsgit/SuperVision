import os
import json
from PIL import Image

import torchvision.transforms as transforms




def get_files(dir=''):
    yield from os.listdir(dir)

def get_cat_mapping(dir=''):
    cat_map = {}
    for entry in json.load(dir)['annotations']:
