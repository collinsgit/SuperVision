import os
import sys

mydir = os.path.abspath(os.path.split(__file__)[0])
cocodir = os.path.abspath(os.path.join(mydir,'../datasets/coco/'))
if not input('Coco is here, right? (y/n) -> %s: ' % cocodir).lower().startswith('y'):
    print("exiting")
    sys.exit()
errors = eval(open(os.path.join(mydir, 'errors.txt')).read())

def remove(f):
    print("Removing",f)
    os.remove(f)

for f in errors:
    bn, ext = os.path.splitext(os.path.basename(f))
    remove(os.path.join(cocodir,'hr', '%s%s' % (bn, ext)))
    remove(os.path.join(cocodir,'lr','X2','%sx2%s' % (bn,ext)))
    remove(os.path.join(cocodir,'lr','X3','%sx3%s' % (bn,ext)))
    remove(os.path.join(cocodir,'lr','X4','%sx4%s' % (bn,ext)))