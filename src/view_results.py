from matplotlib.pyplot import *
import os
import argparse
import math
from PIL import Image
import random
import torchvision.transforms as transforms

src = os.path.abspath(os.path.split(__file__)[0])
cocodir = os.path.abspath(os.path.join(src,'..','datasets','coco'))
hrpath = os.path.join(cocodir, 'hr')
lrpath = os.path.join(cocodir,'lr')

def resize(im, percentage):
    w, h = im.size
    w, h = int(w * (percentage / 100)), int(h * (percentage / 100))
    return im.resize((w,h))

def xor(a,b):
    return a or b and not (a and b)

def model_to_dir(p):
    return os.path.abspath('../experiment/%s/results-Coco' % p)

def parse_args():
    parser = argparse.ArgumentParser(description='Views SR results side by side')
    parser.add_argument('--models', required=True, nargs='+', help="Which models to show results for")
    parser.add_argument('--random', required=False, type=int, default=None,help='Randomly sample this many images')
    parser.add_argument('--scale', required=False, type=int, nargs='+', default=(4,), help="Show results from these scales")
    parser.add_argument('--indices',required=False, type=lambda s: list(map(int,s.split('-'))), nargs='+', default=None)
    parser.add_argument('--ids',required=False, type=int, nargs='+', default=None)
    parser.add_argument('--train', required=False, action='store_const', const='train', default='val', help='Check for this image in the train set instead of validation')
    args = parser.parse_args()
    assert args.ids is not None or args.random is None or args.indices is None, 'You must either pass ids or random'
    
    for p in args.models:
        assert os.path.isdir(model_to_dir(p)), "The provided model directory doesn't exist:\n%s" % (p)
    return args

def plotmany(fnames, horizontallabels, verticallabels):
    fig = figure(0)
    nr = len(fnames) + 1
    nc = max(map(len, fnames)) + 1
    # horizontallabels = model names
    # verticallabels = image names
    for r in range(len(fnames)):
        for c in range(len(fnames[r])):
            model_name, image_name = (horizontallabels[c], verticallabels[r])
            subplot(nr, nc, (r+1) * nc + (c+1) + 1)
            axis('off')
            im = Image.open(fnames[r][c])
            im.save('../slides/%s_%s.jpg' % (str(image_name), model_name))
            imshow(im)
    for i in range(len(horizontallabels)):
        subplot(nr, nc, i+2)
        axis('off')
        text(0.5,0.5,horizontallabels[i],horizontalalignment='center', verticalalignment='center')
    for i in range(len(verticallabels)):
        subplot(nr, nc, (i+1) * nc + 1)
        axis('off')
        text(0.5,0.5,verticallabels[i],horizontalalignment='center',verticalalignment='center')
    show()

def get_ids(args):
    ids_to_show = set()
    allhr = os.listdir(os.path.join(cocodir, args.train, 'hr'))
    allhrids = set([int(os.path.splitext(q)[0]) for q in allhr])
    poss = allhrids
    for m in args.models:
        poss.intersection_update(set([int(p.split('_')[0]) for p in os.listdir(model_to_dir(m))]))
    if args.indices is not None:
        for idrange in args.indices:
            assert len(idrange) > 0, 'The provided range %s is invalid' % str(idrange)
            ids_to_show.update(set(allhrids[idrange[0]: idrange[-1]]))
    allhrids = sorted(poss)
    if args.ids is not None:
        ids_to_show.update(set(args.ids))
    if args.random is not None:
        random.shuffle(allhrids)
        num_added = 0
        for drawn in allhrids:
            if drawn not in ids_to_show:
                ids_to_show.add(drawn)
                num_added += 1
                if num_added >= args.random:
                    break
    return sorted(ids_to_show)

def get_fn(args, model, imid, scale):
    if model == 'hr':
        return os.path.join(cocodir, args.train, 'hr', '%012d.jpg' % (imid))
    elif model == 'lr':
        return os.path.join(cocodir, args.train, 'lr','X%d' % scale,'%012dx%d.jpg' % (imid,scale))
    else:
        return '%s/%012d_x%d_SR.png' % (model_to_dir(model), imid, scale)

def get_fnames(args):
    # First, let's get the ids we want
    ids_to_show = get_ids(args)
    fnames = []
    models = ['hr','lr']
    models.extend(args.models)
    vert = list(map(str,ids_to_show))
    hor = list(map(str, models))
    for pic in ids_to_show:
        fnames.append([])
        for model in models:
            for scale in args.scale:
                fnames[-1].append(get_fn(args, model, pic, scale))
    return fnames, hor, vert


def main():
    args = parse_args()
    fnames, h, v = get_fnames(args)
    plotmany(fnames, h, v)


if __name__ == "__main__":
    main()

