import numpy as np
import tensorflow as tf


def predict(input_data: np.array) -> np.array:
    model = tf.keras.models.load_model("../model/CNN-best")
    X_test = input_data.reshape(-1, 56, 56, 1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_pred = y_pred.reshape(-1, 1)
    return y_pred
