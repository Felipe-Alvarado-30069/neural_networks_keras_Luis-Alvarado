import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

def main():
    # Cargar el dataset MNIST
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    print(train_data_x.shape)
    print(train_labels_y[1])
    plt.imshow(train_data_x[2000])
    print(test_data_x.shape)
    plt.show()

    # Arquitectura de la red
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compilación
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Resumen del modelo
    model.summary()

    # Preprocesamiento de datos
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    y_train = to_categorical(train_labels_y)

    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
    y_test = to_categorical(test_labels_y)

    # Entrenar el modelo
    model.fit(x_train, y_train, epochs=5, batch_size=128)

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

if __name__ == "__main__":
    main()

