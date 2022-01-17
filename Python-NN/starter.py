import tensorflow as tf
import numpy as np
from tensorflow import keras
# Please place in your report:
# 1. Code
# 2. Model summary: model.summary()
# 3. Predicted values: 
# pred = np.array([10, 20])
# print(model.predict(pred))
# 4. Model weights: print(model.layers[0].get_weights())
# - compare model weights with linear equation of ys = f(xs)

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
    ])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict((10.0,), batch_size=1))

w = model.layers[0].get_weights()
print(w)

lin_eq = lambda x: w[0]*x + w[1]*1

print(lin_eq(10.0))
print(model.summary())
