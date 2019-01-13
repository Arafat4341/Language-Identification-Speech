import os
import sys
import html
import time
import uuid
import argparse
import subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd
from yaml import load

import tensorflow
import tensorflow.keras
from tensorflow.keras.models import load_model

from google_apis import text_to_speech, transcribe_speech, translate_text

lib_dir = os.path.join(os.getcwd(), "./tensorflow")
sys.path.append(lib_dir)
from SpectrogramGenerator import SpectrogramGenerator

lib_dir = os.path.join(os.getcwd(), "./data")
sys.path.append(lib_dir)
from normalise import neg23File

tensorflow.keras.backend.clear_session()
# Path to the the trained model
model_dir = './weights.07.model'
print("Loading model: {}".format(model_dir))
model = load_model(model_dir)
#----------------------------------------------------
# A necessary step if the model was trained using multiple GPUs.
# Adjust parameters if you used different ones while training
optimizer = tensorflow.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=optimizer, 
              loss="categorical_crossentropy", 
              metrics=["accuracy"]) 
print("Model compiled.")
#----------------------------------------------------

# https://github.com/keras-team/keras/issues/6462#issuecomment-385962748
global graph
graph = tensorflow.get_default_graph()


def predict(input_file):

    config = load(open('tensorflow/config.yaml', "rb"))
    class_labels = config["label_names"]
    
    params = {"pixel_per_second": config["pixel_per_second"], "input_shape": config["input_shape"], "num_classes": config["num_classes"]}
    data_generator = SpectrogramGenerator(input_file, params, shuffle=False, run_only_once=True).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    # https://github.com/keras-team/keras/issues/6462
    with graph.as_default():
        probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(classes, class_labels[average_class], average_prob)
    return class_labels[average_class]


def count_down():
    print('ll start recording in:')
    countdown = 3
    while (countdown):
        print(countdown)
        countdown -=  1
        time.sleep(0.7)
    print('Recording!')

def downsample_normalise(filename, downsampled):
    cmd = ["ffmpeg", "-i", filename, "-map", "0", "-ac", "1", "-ar", "16000", downsampled]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    neg23File(downsampled)

def play(audio_file_path):
    cmd = ["ffplay", "-nodisp", "-autoexit", audio_file_path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()

if __name__ == "__main__":

    fs = 44100 #Hz
    duration = 3 #seconds

    to_record = 'y'
    while to_record == 'y':

        count_down()
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        print('Finished recording')

        guid = uuid.uuid4()
        filename = 'recording_{0}.wav'.format(guid)
        sf.write(filename, recording, fs)
        
        # downsampling for the purpose of predicting the language
        # audio normalisation
        downsampled = 'downsampled_{0}.wav'.format(guid)
        downsample_normalise(filename, downsampled)
        
        print('\nIdentifying the language...')
        print('The proportions below are in the following order: Croatian, French, Spanish')
        result = predict(downsampled)

        try:
            transcribed = transcribe_speech(filename, result)
            translated = translate_text(transcribed)
            text_to_speech(translated)
            print('Playing the translation!')
            play('output.mp3')
            os.remove('output.mp3')
        except Exception as e:
            print('\nThe following exception occured:')
            print(e)
            print("--------------------------\n \
            Note from the author: If it's not a Google API error, \
            it may just be that the sounddevice wasn't able to access the microphone\n \
            --------------------------")
        os.remove(filename)
        os.remove(downsampled)

        print('\nWould you like to record and identify again? [y/n]: ')
        to_record = input()
    