import io
import os
import sys
import html
import time
import uuid
import argparse
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

lib_dir = os.path.join(os.getcwd(), "./tensorflow")
sys.path.append(lib_dir)

from SpectrogramGenerator import SpectrogramGenerator

tensorflow.keras.backend.clear_session()
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
        print('Audio content written to file "output.mp3"')


fs = 44100
sd.default.samplerate = fs
duration = 3  # seconds

to_record = 'y'
while to_record == 'y':
    print('In three seconds I\'ll start recording!')
    count_down = 3
    while (count_down):
        print(count_down)
        count_down -=  1
        time.sleep(1)

    print('Recording!')
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print('Finished recording')

    filename = 'recording_{0}.wav'.format(uuid.uuid4())
    sf.write(filename, recording, fs)

    print('Identifying the language...')
    result = predict(filename)
    print(result)
    try:
        transcribed = transcribe_file(filename, lang_codes[result])
        translated = translate_text(transcribed)
        text_to_speech(translated)
    except UnboundLocalError as e:
        print('\nThe following exception occured:')
        print(e)
        print("It is most likely that the microphone didn't catch any sound.")
    #os.remove(filename)

    print('\nWould you like to record and identify again? [y/n]: ')
    to_record = input()
   