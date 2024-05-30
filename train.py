from keras.models import load_model
import numpy as np

# Load preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Load the model
model = load_model('lstm_model.h5')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save('lstm_model_trained.h5')
