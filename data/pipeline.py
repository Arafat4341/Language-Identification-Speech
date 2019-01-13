import os
import subprocess
import argparse
import shutil
from yaml import load

from ExtendedDict import ExtendedDict
from download_youtube import download_user_and_playlist
from normalise import normalise
from wav_to_spectrogram import directory_to_spectrograms
from delete_bad_images import delete_bad_images
from organise_spectograms import organise

config = load(open('../tensorflow/config.yaml', "rb"))

def pipeline(output_path, max_downloads):
    ''' 
    The pipeline does the following:
    - downloads dataset
    - normalises the audio
    - converts audio to spectrograms
    - deletes bad spectrograms
    - organises the dataset for feeding it to NN
    '''

    segmented = os.path.join(output_path, "segmented")
    unorg_spectrograms = os.path.join(output_path, "unorg_spectrograms")
    org_spectrograms = os.path.join(output_path, "org_spectrograms")

    print('Youtube download started!')
    download_user_and_playlist(output_path, max_downloads)
    # deletes the folder with the unsegmented audio files
    shutil.rmtree(os.path.join(output_path, "raw"))

    # normalisation of segmented audio files
    print('\nLoudness normalisation of segmented audio files started!\n')
    args = ExtendedDict( {
        'source': segmented
    } )
    normalise(args)

    # segmented wav files to spectrograms
    print('\nConversion of segmented wav files to spetrograms started!\n')
    args = ExtendedDict( {
        'shape': config['input_shape'],
        'pixel_per_second': config['pixel_per_second'],
        'languages': config['label_names'],
        'source': segmented,
        'target': unorg_spectrograms
    } )
    directory_to_spectrograms(args)

    # delete corrupted spectrograms
    print('\nDeletion of bad spectrogram images started!\n')
    args = ExtendedDict( {'source': unorg_spectrograms} )
    delete_bad_images(args)

    # organise spectrograms by their classes and training, validation and test sets
    # in order to use the ImageDataGenerator's flow_from_directory()
    print('Started organising spectrograms!')
    args = ExtendedDict( {
        'source': unorg_spectrograms,
        'target': org_spectrograms
    } )
    organise(args)

    shutil.rmtree(unorg_spectrograms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', dest='output_path', default=os.getcwd(), required=True)
    parser.add_argument('--downloads', dest='max_downloads', default=1200)
    args = parser.parse_args()

    pipeline(args.output_path, args.max_downloads)