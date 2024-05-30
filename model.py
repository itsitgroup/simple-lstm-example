from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
import tensorflow as tf

# Load preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Define the model
input = Input(shape=(100,))
x = Embedding(input_dim=5000, output_dim=128, input_length=100)(input)
x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.5)(x)
x = LSTM(128)(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model architecture
model.save('lstm_model.h5')
