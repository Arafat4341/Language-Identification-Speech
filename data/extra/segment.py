'''
I used this script for the sound files that I already had downloaded and 
only wanted to segment them. This script is basically the same as 
the download_youtube.py script but without the part for downloading
audio files.
'''

import subprocess
import os
import sys
import argparse
import glob
import string
import yaml
from collections import Counter

lib_dir = os.path.join(os.getcwd(), "../")
sys.path.append(lib_dir)

from create_csv import create_csv

file_counter = Counter()


def clean_filename(filename):
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    new_name = "".join(c for c in filename if c in valid_chars)
    new_name = new_name.replace(' ','_')
    return new_name


def segment():

    raw_path = os.path.abspath(args.raw_path)
          
    files = glob.glob(os.path.join(raw_path, "*.wav"))

    segmented_path = os.path.abspath(args.segmented_path)
    segmented_files = glob.glob(os.path.join(segmented_path, "*.wav"))
    for f in files:

        cleaned_filename = clean_filename(os.path.basename(f))
        cleaned_filename = cleaned_filename[:-4]

        output_filename = os.path.join(segmented_path, cleaned_filename + "_%03d.wav")

        command = ["ffmpeg", "-y", "-i", f, "-map", "0", "-ac", "1", "-ar", "16000", "-f", "segment", "-segment_time", "3", output_filename]
        subprocess.call(command)

    print('Done')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', dest='raw_path', default=os.getcwd(), required=True)
    parser.add_argument('--segmented', dest='segmented_path', default=os.getcwd(), required=True)

    args = parser.parse_args()

    segment()

    print(file_counter)
