# Record and Predict the Language

Finally, you may want to test your trained model on your own voice, and more importantly, live. Here, you'll be able to record, predict the language and, if desired, get the translation of it using Google APIs.

By running

    python live_predict_translate.py --model <trained_model_path>

you'll be able to record 3 seconds of your voice and get the prediction of the language in which you spoke.

If you'd like to use Google APIs in order to get the translation, you should create an account on the Google Cloud Platform, enable the the needed APIs and define the path to your credentials in the `google_apis.py` and run

    python live_predict_translate.py --model <trained_model_path> --google_apis 1

## Web App
 
 If you'd like to have your trained model running as a web app, check my another repo - [The Babel Fish v0.01](https://github.com/ibro45/Speech-to-Speech-Translator)