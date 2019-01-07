from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Permute, Reshape, Bidirectional, LSTM

NAME = "InceptionResNetV2 CRNN"

def create_model(input_shape, num_classes):

    input_tensor = Input(shape=input_shape)  
    inceptionresnetv2_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=input_tensor)

    x = inceptionresnetv2_model.output
    #x = GlobalAveragePooling2D()(x)

    # (bs, y, x, c) --> (bs, x, y, c)
    x = Permute((2, 1, 3))(x)

    # (bs, x, y, c) --> (bs, x, y * c)
    _x, _y, _c = [int(s) for s in x.shape[1:]]
    x = Reshape((_x, _y*_c))(x)
    x = Bidirectional(LSTM(512, return_sequences=False), merge_mode="concat")(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inceptionresnetv2_model.input, outputs=predictions)

    return model
    