from keras.models import load_model
import numpy as np

# Load preprocessed data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load the trained model
model = load_model('lstm_model_trained.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
