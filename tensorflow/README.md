Once you have your dataset ready, you're ready for training the neural network.
The usual steps you should take are (the steps that are in bold are mandatory):

 1. Define the model inside the `models/` folder (skip if you want to use the proposed architecture)
 2. **Specify the parameters in the** `config.yaml`
 3. If desired, change the optimiser or its parameters in the `compile_model.py`
 4. **Train the model by running:** 
 `python train.py`
 6. **Evaluate it on your test set with:** 
 `python evaluate.py --model <trained_model_path>`

## Details about the scripts and files

| File | Description |
|--|--|
| `config.yaml` | Contains details important for training the neural network.|
| `SpectrogramGenerator.py` | Takes audio files and creates spectrograms out of them. |
| `compile_model.py` | Defines the optimiser and its parameters for compiling the neural network. |
| `train.py` | Trains the neural network. The results are saved in the folder `logs/<date-time-it-was-ran>/` |
| `evaluate.py` | Evaluates the trained model of the NN. Outputs the accuracy and a confusion matrix. |
| `predict.py` | Uses the trained model to predict the language of an audio file. Outputs the predicted language and the proportions of its confidence for each of the languages. |


