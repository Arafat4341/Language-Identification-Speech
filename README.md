# Language Identification From Speech

The repository contains the code that allows you to obtain a dataset for desired languages and train a neural network that's able to identify the language spoken in a recording as short as 3 seconds.

This code follows my bachelor's thesis, *"Recognition of Natural Language Type Using Tensorflow"*, where you can find the detailed description of the project along with the results achieved.

I'd also like to acknowledge that this project was based on the [project](https://github.com/HPI-DeepLearning/crnn-lid) that this repository has forked from and their paper, [*"Language Identification Using Deep Convolutional Recurrent Neural Networks"*](https://arxiv.org/abs/1708.04811). 


## Repository Structure
The repository is separated into four categories, with each containing its own ReadMe. 

| Folder | Description |
|--|--|
| [*data/*](https://github.com/ibro45/Language-Identification-Speech/tree/master/data) | Contains the scripts for downloading and preprocessing the dataset. |
| [*tensorflow/*](https://github.com/ibro45/Language-Identification-Speech/tree/master/tensorflow) | Neural-network-related scripts. Creation, training, evaluation etc.|
| [*record_predict/*](https://github.com/ibro45/Language-Identification-Speech/tree/master/record_predict) | Scripts that allow the user to record his/her speech and test it with the trained model.|
| [*paper/*](https://github.com/ibro45/Language-Identification-Speech/tree/master/paper)| Contains the author's paper titled *"Recognition of Natural Language Type Using Tensorflow"*.|

## Requirements
To install all the required packages, run

    pip install -r requirements.txt

Please note that the `requirements.txt` specifies the CPU-based `tensorflow`. If you'd like to train the neural network using GPU(s), check `tensorflow-gpu`. Note that the installation of it can be tricky, but using `conda` and this [tutorial](https://www.pugetsystems.com/labs/hpc/Install-TensorFlow-with-GPU-Support-the-Easy-Way-on-Ubuntu-18-04-without-installing-CUDA-1170/) may make it a bit easier. Also, make sure that you have the version of Nvidia drivers compatible with the current `tensorflow-gpu` version.

Furthermore, you should have the following utilities installed in your system:

- FFmpeg
- SoX
- youtube-dl


If you're running Ubuntu, you can install them by running:

    sudo apt install ffmpeg sox youtube-dl



