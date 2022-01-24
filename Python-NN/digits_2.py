import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data


# Synthetic handwriting 
files = [(str(n) + ".png") for n in range(10)]
x_realTest = []
for file in files:
    x_real=keras.preprocessing.image.load_img(file,target_size=(28,28),color_mode="grayscale")
    x_realTest.append(keras.preprocessing.image.img_to_array(x_real))

for i in range(10):
	plt.subplot(2, 5, i+1)
	plt.imshow(x_realTest[i], cmap=plt.get_cmap('gray'))
plt.show()

# Real handwriting (A)
files = [(str(n) + "B.png") for n in range(10)]
x_realTest_A = []
for file in files:
    x_real=keras.preprocessing.image.load_img(file,target_size=(28,28),color_mode="grayscale")
    x_realTest_A.append(keras.preprocessing.image.img_to_array(x_real))

for i in range(10):
	plt.subplot(2, 5, i+1)
	plt.imshow(x_realTest_A[i], cmap=plt.get_cmap('gray'))
plt.show()

# Real handwriting (B)
files = ["./blured/" + (str(n) + "B.png") for n in range(10)]
x_realTest_B = []
for file in files:
    x_real=keras.preprocessing.image.load_img(file,target_size=(28,28),color_mode="grayscale")
    x_realTest_B.append(keras.preprocessing.image.img_to_array(x_real))

for i in range(10):
	plt.subplot(2, 5, i+1)
	plt.imshow(x_realTest_B[i], cmap=plt.get_cmap('gray'))
plt.show()


(x_train, y_train), (x_test, y_test) = load_data()  # MNIS Dataset

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

for i in range(25):
	plt.subplot(5, 5, i+1)
	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

in_shape = x_train.shape[1:]

n_classes = len(np.unique(y_train))
print(in_shape, n_classes)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
# model.add(keras.layers.MaxPool2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(n_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %.3f' % acc)

image = x_test[np.random.randint(0, x_test.shape[0]-1)]

yhat = model.predict(np.asarray([image]))
print('Predicted: class=%d' % np.argmax(yhat))

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()


# Predictions for dataset digits
for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(x_test[i], cmap=plt.get_cmap('gray'))
    yh = model.predict(np.asarray([x_test[i]]))
    pred = np.argmax(yh)
    ax.set_title('Is=' + str(y_test[i]) + ' pred=' + str(pred), fontsize=10)
plt.show()


# Predictions for my digits - synthetic
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(x_realTest[i], cmap=plt.get_cmap('gray'))
    yh = model.predict(np.asarray([x_realTest[i]]))
    pred = np.argmax(yh)
    ax.set_title('Ireal=' + str(i) + ' pred=' + str(pred), fontsize=10)
plt.show()

# Predictions for my digits - handwriting similar to MNIS database (A)
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(x_realTest_A[i], cmap=plt.get_cmap('gray'))
    yh = model.predict(np.asarray([x_realTest_A[i]]))
    pred = np.argmax(yh)
    ax.set_title('Ireal_A=' + str(i) + ' pred=' + str(pred), fontsize=10)
plt.show()


# Predictions for my digits - handwriting similar to MNIS database (B)
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(x_realTest_B[i], cmap=plt.get_cmap('gray'))
    yh = model.predict(np.asarray([x_realTest_B[i]]))
    pred = np.argmax(yh)
    ax.set_title('Ireal_B=' + str(i) + ' pred=' + str(pred), fontsize=10)
plt.show()


model.summary()
keras.utils.plot_model(model, 'model_2.png', show_shapes=True)