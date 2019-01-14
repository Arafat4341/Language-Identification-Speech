# -*- coding: utf-8 -*-

import os
import argparse
import scipy.misc
import numpy as np
import sys
import imageio

from yaml import load

lib_dir = os.path.join(os.getcwd(), "../tensorflow")
sys.path.append(lib_dir)

from SpectrogramGenerator import SpectrogramGenerator
from create_csv import create_csv

def directory_to_spectrograms(args):

    source = args.source
    config = {
        "pixel_per_second": args.pixel_per_second,
        "input_shape": args.shape
    }

    # Start a spectrogram generator for each class
    # Each generator will scan a directory for audio files and convert them to spectrogram images
    # adjust this if you have other languages or any language is missing
    
    languages = args.languages
    

    generators = [SpectrogramGenerator(os.path.join(source, language), config, shuffle=False, run_only_once=True) for language in languages]
    generator_queues = [SpectrogramGen.get_generator() for SpectrogramGen in generators]

    for language in languages:
        output_dir = os.path.join(args.target, language)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    i = 0
    while True:

        target_shape = tuple(args.shape)

        try:
            for j, language in enumerate(languages):
                data = next(generator_queues[j])

                assert data.shape == target_shape, "Shape mismatch {} vs {}".format(data.shape, args.shape)

                file_name = os.path.join(args.target, language, "{}.png".format(i))
                imageio.imwrite(file_name, np.squeeze(data))

            i += 1

            if i % 1000 == 0:
                print("Processed {} images".format(i))

        except StopIteration:
            print("Saved {} images. Stopped on {}".format(i, language))
            break

    create_csv(args.target)

if __name__ == "__main__":
    
    config = load(open('../tensorflow/config.yaml', "rb"))

    # default values, you can still change them by specifying different values when executing the script
    shape = config['input_shape']
    pixel_per_second = config['pixel_per_second']
    languages = config['label_names']

    parser = argparse.ArgumentParser()
    # have default values, can be specified if desired
    parser.add_argument('--shape', dest='shape', default=shape, type=int, nargs=3)
    parser.add_argument('--pixel', dest='pixel_per_second', default=pixel_per_second, type=int)
    parser.add_argument('--languages', dest='languages', default=languages)
    # you have to specify these when executing the script
    parser.add_argument('--source', dest='source', required=True)
    parser.add_argument('--target', dest='target', required=True)
    cli_args = parser.parse_args()

    directory_to_spectrograms(cli_args)


