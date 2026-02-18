import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train[..., None]/255.0, X_test[..., None]/255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)

print("Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])
