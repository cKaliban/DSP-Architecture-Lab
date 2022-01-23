import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data


# Comparison with test values and predicted values
# Put your handwritten digits into the system and test
(x_train, y_train), (x_test, y_test) = load_data() #    MNIS Dataset