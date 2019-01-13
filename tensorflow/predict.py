import argparse
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from yaml import load

from SpectrogramGenerator import SpectrogramGenerator


def predict(cli_args):

    config = load(open(cli_args.config, "rb"))
    class_labels = config["label_names"]
    
    # the file is not normalised before predicting in this script
    params = {"pixel_per_second": config["pixel_per_second"], "input_shape": config["input_shape"], "num_classes": config["num_classes"]}
    data_generator = SpectrogramGenerator(cli_args.input_file, params, shuffle=False, run_only_once=True).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    # Model Generation
    model = load_model(cli_args.model_dir)
    #----------------------------------------------------
    # A necessary step if the model was trained using multiple GPUs.
    # Adjust parameters if you used different ones while training
    optimizer = tensorflow.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(optimizer=optimizer, 
                loss="categorical_crossentropy", 
                metrics=["accuracy"]) 
    print("Model compiled.")
    #----------------------------------------------------

    probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(classes, class_labels[average_class], average_prob)
    return probabilities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--input', dest='input_file', required=True)
    parser.add_argument('--config', dest='config', default="config.yaml")
    cli_args = parser.parse_args()

    if not os.path.isfile(cli_args.input_file):
        sys.exit("Input is not a file.")


    predict(cli_args)
