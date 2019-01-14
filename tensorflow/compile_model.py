import tensorflow
from yaml import load 

def compile_model(model):
    config = load(open('../tensorflow/config.yaml', "rb"))
    
    optimizer = tensorflow.keras.optimizers.Adam(lr=config["learning_rate"], decay=1e-6)
    model.compile(optimizer=optimizer, 
                loss="categorical_crossentropy", 
                metrics=["accuracy"]) 
    print("Model compiled.")
    #----------------------------------------------------
    return model