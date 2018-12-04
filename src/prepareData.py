import os
import shutil
import argparse
import PIL
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets
from tqdm import tqdm
from data.coco import CocoClasses
import torch.multiprocessing
import numpy as np
from data import embedimage

# Directory Structure:

# SuperVision
# |> datasets
# |---> coco
# |------> annotations
# |------> train
# |---------> avg_embeddings
# |---------> hr
# |---------> lr
# |------------> X2
# |------------> X3
# |------------> X4
# |------> val
# |---------> avg_embeddings
# |---------> hr
# |---------> lr
# |------------> X2
# |------------> X3
# |------------> X4

# Top level directories
srcdir = os.path.split(__file__)[0]
datasetdir = os.path.abspath(os.path.join(srcdir,'../datasets'))

# Coco directories
cocodir = os.path.join(datasetdir,'coco')
annotationsdir = os.path.join(cocodir,'annotations')
annotations_val_file = os.path.join(annotationsdir, 'instances_val2017.json')
annotations_train_file = os.path.join(annotationsdir, 'instances_train2017.json')

# Training directories
cocotraindir = os.path.join(cocodir, 'train')
cocotrainhr = os.path.join(cocotraindir, 'hr')
cocotrainlr = os.path.join(cocotraindir, 'lr')

cocotrainavg = os.path.join(cocotraindir, 'avg_embeddings')

# Validation directories
cocovaldir = os.path.join(cocodir, 'val')
cocovalhr = os.path.join(cocovaldir, 'hr')
cocovallr = os.path.join(cocovaldir, 'lr')
cocovalavg = os.path.join(cocovaldir, 'avg_embeddings')


def validate_directories(scales):
    # Coco
    os.makedirs(datasetdir,exist_ok=True)
    os.makedirs(cocodir,exist_ok=True)

    # Find annotations
    assert os.path.isdir(annotationsdir), "The annotations directory could not be found:\n %s " % (annotationsdir)
    assert os.path.isfile(annotations_train_file), "Could not find training annotations at: %s "% annotations_train_file
    assert os.path.isfile(annotations_val_file), "Could not find validation annotations at: %s "% annotations_val_file
    
    # Training directories
    os.makedirs(cocotraindir,exist_ok=True)
    os.makedirs(cocotrainhr, exist_ok=True)
    os.makedirs(cocotrainlr, exist_ok=True)
    for scale in scales:
        os.makedirs(os.path.join(cocotrainlr, 'X%d' % scale),exist_ok=True)
    os.makedirs(cocotrainavg, exist_ok=True)
    
    # Validation directories
    os.makedirs(cocovaldir,exist_ok=True)
    os.makedirs(cocovalhr, exist_ok=True)
    os.makedirs(cocovallr, exist_ok=True)
    for scale in scales:
        os.makedirs(os.path.join(cocovallr, 'X%d' % scale),exist_ok=True)
    os.makedirs(cocovalavg, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Filters, Crops, and Resizes training/validation images")
    parser.add_argument('unzipped_train_dir',help='Where the original training images are located')
    parser.add_argument('unzipped_val_dir',help='Where the original validation images are located')
    parser.add_argument('--scales',required=False,default=(2,3,4), nargs='+',type=int,help='Which LR scales to use')
    parser.add_argument('--unzipped_annotation_dir',required=False, type=os.path.abspath,help='Where the original annotation files are located (will be copied)')
    parser.add_argument('--batch_size', required=False, default=16, type=int)
    parser.add_argument('--skip_train', required=False, action='store_const', const=True, default=False,help="Skip processing the training data")
    parser.add_argument('--skip_val', required=False, action='store_const', const=True, default=False,help="Skip processing the validation data")
    parser.add_argument('--skip_rsz', required=False, action='store_const', const=True, default=False,help="Skip cropping/resizing")
    parser.add_argument('--skip_avg', required=False, action='store_const', const=True, default=False,help="Skip computing categorical averages")
    args = parser.parse_args()
    return args


def copy_annotations(unzipped_annotation_dir):
    assert os.path.isdir(unzipped_annotation_dir),"Provided annotations directory is not a directory:\n%s" % (unzipped_annotation_dir)
    os.makedirs(annotationsdir, exist_ok=True)
    original_annotations_val_file = os.path.join(unzipped_annotation_dir, 'instances_val2017.json')
    original_annotations_train_file = os.path.join(unzipped_annotation_dir, 'instances_train2017.json')
    assert os.path.isfile(original_annotations_val_file), \
        "Validation annotations file could not be found!"
    assert os.path.isfile(original_annotations_train_file), \
        "Training annotations file could not be found!"
    print("Copying Train/Validation annotation files...")
    for cpargs in tqdm([
                    (original_annotations_val_file, annotations_val_file), 
                    (original_annotations_train_file, annotations_train_file)]
                    ):
        shutil.copy(*cpargs)


class CategoryDataSet(torch.utils.data.Dataset):
    def __init__(self, valid_files, categories, category_id, roothr, *args, **kwargs):
        self.files = list(set(categories.get_files_for_category(category_id)).intersection(valid_files))
        self.tsfm = transforms.ToTensor()
        self.roothr = roothr
        self.category_id = category_id
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
         return self.tsfm(Image.open(os.path.join(self.roothr,self.files[idx])))

    def __len__(self):
        return len(self.files)


def compute_average_for_category(ds, em, device, output_dir, batch_size=16):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    total_results = None
    num_batches = len(loader)
    if len(ds) == 0:
        return
    print("Averaging Category: %2d" % (ds.category_id))
    for im in tqdm(loader):
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
    total_results *= 1.0 / len(ds)
    total_results = total_results.astype(np.float32)
    torch.save(total_results, os.path.join(output_dir, "%02d.pt" % ds.category_id))


def _process_image(args):
    fname, imagedir, outputhr, outputlr, scales, desired_size, batch_size = args
    orig_fname = os.path.join(imagedir, fname)
    basename, ext = os.path.splitext(fname)
    im = Image.open(orig_fname).convert("RGB")
    dim = min(im.size)
    if dim < desired_size:
        return 0
    tsfm = transforms.Compose([
        transforms.CenterCrop(dim),
        transforms.Resize((desired_size, desired_size))
    ])
    im = tsfm(im)
    im.save(os.path.join(outputhr,fname))
    for scale in scales:
        tsfm_smaller = transforms.Resize((desired_size // scale, desired_size // scale))
        im_small = tsfm_smaller(im)

        im_small.save(os.path.join(outputlr, "X%d" % scale, "%sx%d%s" % (basename,scale,ext)))
        del im_small
        del tsfm_smaller
    del im
    del tsfm
    return 1


def do_image_resizing(found_and_labelled_files, imagedir, categories, 
            outputhr, outputlr, outputavg, scales, 
            desired_size=300, n_workers=6, batch_size=16):
    
    print("Cropping and Resizing all images. This will take some time!")
    pool = torch.multiprocessing.Pool(processes=n_workers)
    pool.map(_process_image,[(fname, imagedir,
            outputhr, outputlr, scales, 
            desired_size, batch_size) for fname in found_and_labelled_files])
    print("All files cropped and resized!")


def do_category_averaging(found_and_labelled_files, imagedir, categories, 
    outputhr, outputlr, outputavg, scales, 
    desired_size=300, n_workers=6, batch_size=16):

    print("Calculating average for each category")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    em = embedimage.VGG()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(),"GPUs")
        em = nn.DataParallel(em)
    em = em.to(device)

    for category_id in categories.get_all_categories():
        ds = CategoryDataSet(found_and_labelled_files, categories, category_id, outputhr)
        compute_average_for_category(ds, em, device, outputavg, batch_size)


def process_images(imagedir, categories, 
            outputhr, outputlr, outputavg, scales, 
            desired_size=300, n_workers=6, batch_size=16, skip_avg=False, skip_crop=False):
    labelled_files = set(categories.get_all_filenames())
    found_files = set(os.listdir(imagedir))
    found_and_labelled_files = labelled_files.intersection(found_files)
    if skip_crop:
        print("Skipping Cropping/Resizing images")
    else:
        do_image_resizing(found_and_labelled_files, imagedir, categories, 
            outputhr, outputlr, outputavg, scales, 
            desired_size, n_workers, batch_size)
    found_files = set(os.listdir(outputhr))
    found_and_labelled_files = labelled_files.intersection(found_files)
    if skip_avg:
        print("Skipping computing averages")
    else:
        do_category_averaging(found_and_labelled_files, imagedir, categories, 
            outputhr, outputlr, outputavg, scales, 
            desired_size, n_workers, batch_size)
    
    return True


def main():
    # args: unzipped_train_dir, unzipped_val_dir, unzipped_annotation_dir=None
    args = parse_args()
    # Copy annotations from original to expected destination
    if args.unzipped_annotation_dir is not None and args.unzipped_annotation_dir != annotationsdir:
        # Annotations are somewhere different at the moment
        copy_annotations(args.unzipped_annotation_dir)
        
    print("Checking directory structure...")
    # Make sure all the directories that we want exist
    validate_directories(args.scales)

    print("Parsing annotation files...")
    train_categories = CocoClasses(annotations_train_file)
    val_categories = CocoClasses(annotations_val_file)

    if args.skip_train:
        print("Skipping Training Data Processing")
    else:
        print("Processing Training Data...")
        process_images(args.unzipped_train_dir, train_categories, cocotrainhr, cocotrainlr, cocotrainavg, args.scales, skip_avg=args.skip_avg, skip_crop=args.skip_rsz, batch_size=args.batch_size)

    if args.skip_val:
        print("Skipping Validation Data Processing")
    else:
        print("Processing Validation Data...")
        process_images(args.unzipped_val_dir, val_categories, cocovalhr, cocovallr, cocovalavg, args.scales, skip_avg=args.skip_avg, skip_crop=args.skip_rsz, batch_size=args.batch_size)


if __name__ == "__main__":
    main()