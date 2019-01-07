'''
This script was created because I wanted to merge two datasets that
contained different files with same names. To avoid overwriting them
I ran this script inside of one dataset which renames each file with
a GUID.
'''

import os
import argparse
import glob
import uuid


def rename(args):
    source = os.path.abspath(args.source)
    for filename in glob.glob(source + '/**/*.png', recursive=True):
        directory, file = filename.rsplit('/', 1)
        f, f_format = file.split('.')

        # new filename
        f = str(uuid.uuid4())
        
        new_filename = directory + '/' + f + '.' + f_format
        os.rename(filename, new_filename)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source', required=True)
    args = parser.parse_args()
    
    rename(args)