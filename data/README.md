# Obtaining and Preprocessing the Data

**Important note**: Define the language names and the number of them in `tensorflow/config.yaml` under `label_names` and `num_classes`.

# The Automated Way

Steps to obtain the dataset:

 1. Define the YouTube channels and playlist you'd like to download from in the `sources.yml`
 2. Run `python pipeline.py --output <path> --downloads <int>`

That's it!

Explanation of the `pipeline.py`'s parameters:

 - `--output` - defines where will the data be downloaded
 - `--downloads` - specifies the number of videos to be downloaded from each playlist/channel. It is an optional parameter and defaults to 1200.

The `pipeline.py` script was created in order to automate the process of getting the dataset and preprocessing it. 
It does the following steps:

 1. Downloads the dataset and segments them into 3-second audio files
 2. Performs the loudness normalisation of the 3-second audio samples
 3. Converts the audio samples to spectrograms 
 4. Deletes the corrupted spectrograms
 5. Organises the spectrograms into training, validation and testing sets, in order to be fed to the neural network using Keras' `ImageDataGenerator` and its [`flow_from_directory`](https://keras.io/preprocessing/image/#flow_from_directory) 

# The Manual Way

If you prefer doing it the hard way and would like to have a bit more of insight into what's happening, you could also do all of the mentioned steps separately.

The scripts are going to be explained in a chronological order, i.e. the order in which you should run them.


### 1. `download_youtube.py` 

This script uses `youtube-dl` for downloading the defined channels and/or playlists and `ffmpeg` for segmenting them into 3 seconds and down-sampling them to 16000 Hz.

Run it by calling:

    python download_youtube.py --output <path> --downloads <int> 

Parameters:

 - `--output` - specifies where will the data be downloaded.
 - `--downloads` - specifies the number of videos to be downloaded from each playlist/channel. It is an optional parameter and defaults to 1200.


 ### 2. `normalise.py`
 
 Performs loudness normalisation. Intended to be done on the segmented audio files.
 
 Run it using the following command:

    python normalise.py --source <path>

Parameter `--source` expects the path to the folder containing segmented audio samples.

### 3. `wav_to_spectrograms.py`

This script converts the segmented audio samples to spectrograms. In order to do it, it uses the `tensorflow/SpectrogramGenerator.py` which relies on `SoX` for generating the spectrograms.

To generate the spectrograms, run:

    python wav_to_spectrograms.py --source <path> --target <path>

Parameters:

 1. `--source` - path to the segmented audio samples.
 2. `--target` - path to the folder where the spectrograms will be generated.

### 4. `delete_bad_images.py`

Sometimes there are spectrograms that contain too much silence. This script removes them from the dataset.

    python delete_bad_images.py --source <path>

Parameter `--source` specifies the directory where spectrograms are located.

### 5. `organise_spectrograms.py`

This scripts moves the spectrograms into a new folder that is compatible with the `ImageDataGenerator`'s  [`flow_from_directory`](https://keras.io/preprocessing/image/#flow_from_directory) .

The script splits the data into training, validation and testing sets and arranges them in such a fashion that makes the generator able to tell to which class a file belongs to by looking at the name of the folder in which it is found.

Run it by by calling:

    python organise_spectrograms.py --source <path> --target <path>


Parameters:

 1. `--source` - path to the generated spectrograms.
 2. `--target` - path to the folder where the spectrograms will be moved and organised.
