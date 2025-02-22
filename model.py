from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from config import IMG_SIZE

def create_lstm_model(input_shape):
    lstm_input = Input(shape=input_shape, name="text_input")  # Named input
    x = LSTM(64, return_sequences=True)(lstm_input)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dense(16, activation='relu')(x)
    return Model(lstm_input, x)

def create_cnn_model(input_shape):
    cnn_input = Input(shape=input_shape, name="image_input")  # Named input
    x = Conv2D(32, (3,3), activation='relu')(cnn_input)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    return Model(cnn_input, x)

def create_fusion_model():
    lstm = create_lstm_model((6,1))
    cnn = create_cnn_model((*IMG_SIZE, 3))
    
    combined = Concatenate()([lstm.output, cnn.output])
    final_output = Dense(1)(combined)
    
    model = Model(inputs={'text_input': lstm.input, 'image_input': cnn.input}, outputs=final_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model


# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
# from config import IMG_SIZE  # Import IMG_SIZE

# def create_lstm_model(input_shape, units=64, dropout=0.2):
#     lstm_input = Input(shape=input_shape)
#     x = LSTM(units, return_sequences=True)(lstm_input)
#     x = Dropout(dropout)(x)
#     x = LSTM(units // 2)(x)
#     x = Dense(16, activation='relu')(x)
#     return Model(lstm_input, x)

# def create_cnn_model(input_shape):
#     cnn_input = Input(shape=input_shape)
#     x = Conv2D(32, (3,3), activation='relu')(cnn_input)
#     x = MaxPooling2D((2,2))(x)
#     x = Flatten()(x)
#     x = Dense(16, activation='relu')(x)
#     return Model(cnn_input, x)

# def create_fusion_model(units=64, dropout=0.2):
#     lstm = create_lstm_model((6,1), units, dropout)
#     cnn = create_cnn_model((*IMG_SIZE, 3))  # Uses IMG_SIZE properly
    
#     combined = Concatenate()([lstm.output, cnn.output])
#     final_output = Dense(1)(combined)
    
#     model = Model(inputs=[lstm.input, cnn.input], outputs=final_output)
#     model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
#     return model
