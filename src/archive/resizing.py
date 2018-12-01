import os
from PIL import Image

import torchvision.transforms as transforms


def get_files(dir=''):
    yield from os.listdir(dir)


def build_hr(size, temp_dir, dest_dir):
    resize = transforms.Resize(size)

    for file in get_files(temp_dir):
        im = Image.open(os.path.join(temp_dir, file))
        dim = min(im.size)
        if dim < size:
            continue

        im = resize(transforms.CenterCrop(dim)(im))
        im.save(os.path.join(dest_dir, file))


def build_lr(size, scales, hr_dir, dest_dir):
    for scale in scales:
        print(scale)

        new_size = int(size / scale)

        resize = transforms.Resize(new_size)

        for file in get_files(hr_dir):
            im = Image.open(os.path.join(hr_dir, file))

            im = resize(im)
            im.save(os.path.join(dest_dir, 'X' + str(scale), ('x' + str(scale) + '.').join(file.split('.'))))


if __name__ == '__main__':
    build_hr(size=300, temp_dir='coco/original', dest_dir='coco/hr')
    build_lr(size=300, scales=(2, 3, 4), hr_dir='coco/hr', dest_dir='coco/lr')
