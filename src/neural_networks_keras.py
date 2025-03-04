# Importación de bibliotecas necesarias
import numpy as np  # Biblioteca para operaciones matemáticas y manipulación de arreglos numéricos
from keras.models import Sequential  # Importación del modelo secuencial de Keras
from keras.layers import Dense, Input  # Capas para construir la red neuronal
from keras.utils import to_categorical  # Utilidad para convertir etiquetas a formato one-hot encoding
from keras.datasets import mnist  # Base de datos MNIST de dígitos escritos a mano
import matplotlib.pyplot as plt  # Biblioteca para la generación de gráficos

# Definición de la función principal para entrenar y evaluar el modelo
def train_and_evaluate():
    """
    Función que carga los datos MNIST, entrena una red neuronal y evalúa su desempeño.
    """

    # Cargar los datos de entrenamiento y prueba desde MNIST
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Imprimir información sobre los datos cargados
    print("Forma de los datos de entrenamiento:", train_data_x.shape)
    print("Etiqueta del primer ejemplo de entrenamiento:", train_labels_y[1])
    print("Forma de los datos de prueba:", test_data_x.shape)
    
    # Visualizar un ejemplo de imagen de entrenamiento
    plt.imshow(train_data_x[1], cmap="gray")
    plt.title("Ejemplo de imagen de entrenamiento")
    plt.show()

    # Normalización de datos:
    # Convertimos las imágenes en vectores de 28x28 (a un solo arreglo de 784 valores)
    # y los normalizamos dividiendo entre 255 (para que estén en el rango [0,1])
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    y_train = to_categorical(train_labels_y)  # Convertimos las etiquetas a formato one-hot

    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
    y_test = to_categorical(test_labels_y)  # Convertimos las etiquetas de prueba a one-hot

    # Definición de la arquitectura de la red neuronal
    model = Sequential([
        Input(shape=(28*28,)),  # Capa de entrada con 784 neuronas (tamaño de la imagen)
        Dense(512, activation='relu'),  # Capa oculta con 512 neuronas y activación ReLU
        Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada dígito 0-9), activación softmax
    ])

    # Compilación del modelo
    model.compile(
        optimizer='rmsprop',  # Optimizador para ajustar los pesos (RMSprop suele funcionar bien para redes profundas)
        loss='categorical_crossentropy',  # Función de pérdida para clasificación multiclase
        metrics=['accuracy']  # Métrica de evaluación (precisión)
    )

    # Entrenamiento del modelo
    model.fit(x_train, y_train, epochs=8, batch_size=128)  # 8 épocas, mini-batch de 128 imágenes

    # Evaluación del modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(x_test, y_test)  # Calcula la pérdida y la precisión en datos de prueba
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")  # Muestra la precisión final

    print("Entrenamiento y evaluación completados exitosamente.")  # Mensaje de finalización

# Evitar ejecución automática si se importa este script en otro módulo
if __name__ == "__main__":
    train_and_evaluate()  # Llama a la función principal si el script se ejecuta directamente
