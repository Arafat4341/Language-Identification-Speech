''' 
This script deletes all bad (completely black) images. After deleting 
those images, it checks the distribution of images between the languages
and makes it even by deleting the needed amount of randomly selected
images for each of the languages. It does so until each language has
the same amount of images as the one that had the least of them initially.
'''

import numpy as np
import imageio
import os
import argparse
import glob
from yaml import load

from create_csv import create_csv

config = load(open('../tensorflow/config.yaml', "rb"))
languages = config["label_names"]

results = dict((lang,0) for lang in languages)

def delete_bad_images(args):
    source = os.path.abspath(args.source)
    for filename in glob.glob(source + '/**/*.png', recursive=True):
        check_image(filename)
    print('Bad spectrogram deletion ended.')
    print('Final number of spectrograms per class and dataset split:')
    create_csv(args.source)

def check_image(image_path):
    image = imageio.imread(image_path, pilmode="L")
    if(np.count_nonzero(image-np.mean(image)) == 0):
        print('Deleted ' + image_path)
        os.remove(image_path)
        results[image_path.split('/')[-2]] += 1 # increments the count for the specific class
        print(results, end='\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source', required=True,
                        help='Path to the folder in which are the language subfolders that contain spectrograms')
    args = parser.parse_args()
    
    delete_bad_images(args)
    
