import io
import os
import sys
import html
import time
import uuid
import argparse
import subprocess
from subprocess import DEVNULL
import numpy as np
import soundfile as sf
import sounddevice as sd
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import load_model
from yaml import load

from google.cloud import texttospeech
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/media/bro/New Volume/Zavrsni rad/Language-Identification-Speech/tensorflow/gcloud_account.json"


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
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]) 
print("Model compiled.")
#----------------------------------------------------

# https://github.com/keras-team/keras/issues/6462
global graph
graph = tensorflow.get_default_graph()

lang_codes = {'croatian': 'hr-HR',
              'french'  : 'fr-FR',
              'spanish' : 'es-ES'}



def predict(input_file):

    config = load(open('tensorflow/config.yaml', "rb"))
    class_labels = config["label_names"]
    
    params = {"pixel_per_second": config["pixel_per_second"], "input_shape": config["input_shape"], "num_classes": config["num_classes"]}
    data_generator = SpectrogramGenerator(input_file, params, shuffle=False, run_only_once=True).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    # Model Generation
    with graph.as_default():
        probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(classes, class_labels[average_class], average_prob)
    return class_labels[average_class]


def transcribe_file(speech_file, language_code): 
    """Transcribe the given audio file."""
    client = speech.SpeechClient()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code=language_code) 

    response = client.recognize(config, audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
    
    return result.alternatives[0].transcript


def translate_text(text):
    # Instantiates a client
    translate_client = translate.Client()
    # Translates some text into Russian
    translation = translate_client.translate(
        text,
        target_language='en')
    print(u'Translation: {}'.format(translation['translatedText']))
    return html.unescape(translation['translatedText'])


def text_to_speech(text, language_code='en-GB'):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)

    # The response's audio_content is binary.
    with open('output.mp3', 'wb') as out:
        # Write the response to the output file.
        out.write(response.audio_content)

def count_down():
    print('ll start recording in:')
    countdown = 3
    while (countdown):
        print(countdown)
        countdown -=  1
        time.sleep(0.7)

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

    fs = 44100
    duration = 3  # seconds

    to_record = 'y'
    while to_record == 'y':
        count_down()
        print('Recording!')
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
            transcribed = transcribe_file(filename, lang_codes[result])
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
            it may just be that an issue with microphone and sounddevice \n \
            --------------------------")
        os.remove(filename)
        os.remove(downsampled)

        print('\nWould you like to record and identify again? [y/n]: ')
        to_record = input()
    