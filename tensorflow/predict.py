import argparse
import numpy as np
import os
import sys
import tensorflow
from yaml import load

from SpectrogramGenerator import SpectrogramGenerator
from load_compile_model import load_compile_model

def predict(model, input_file, config):

    config = load(open(config, "rb"))
    class_labels = config["label_names"]
    
    # the file is not normalised before predicting in this script
    params = {"pixel_per_second": config["pixel_per_second"], "input_shape": config["input_shape"], "num_classes": config["num_classes"]}
    data_generator = SpectrogramGenerator(input_file, params, shuffle=False, run_only_once=True).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(classes, class_labels[average_class], average_prob)
    #return probabilities
    return class_labels[average_class]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--input', dest='input_file', required=True)
    parser.add_argument('--config', dest='config', default="config.yaml")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        sys.exit("Input is not a file.")

    model = load_compile_model(args.model_dir)
    predict(model, args.input_file, args.config)
