# README - Entrenamiento de Red Neuronal con Keras

## Información del Proyecto
**Materia:** Sistemas de Visión Artificial\
**Tarea:** Tarea 2.2_Entrenamiento de red neuronal con Keras\ 
**Estudiante:** Luis Felipe Alvarado Resendez\
**Fecha:** 03/03/2025\

## Descripción General
Este repositorio contiene un código en Python para entrenar una red neuronal utilizando la biblioteca Keras. El modelo se entrena con la base de datos MNIST, que contiene imágenes de dígitos escritos a mano (0-9). El objetivo del proyecto es diseñar una red neuronal capaz de reconocer estos dígitos con alta precisión.

## Requisitos Previos
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas en tu entorno de Python:
```bash
pip install numpy keras matplotlib
```

## Estructura del Repositorio
```
📂 proyecto_red_neuronal/
├── 📂 src/                      # Código fuente
│   ├── neural_networks_keras.py # Script principal con la implementación de la red neuronal
├── main.py                      # Punto de entrada para ejecutar el entrenamiento
├── README.md                    # Este documento
```

## Explicación del Código

### 1. Carga de Datos
El código utiliza la base de datos MNIST, que contiene 60,000 imágenes para entrenamiento y 10,000 para prueba. Estas imágenes son de 28x28 píxeles y representan números del 0 al 9.
```python
(train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
```

### 2. Preprocesamiento de Datos
Para que la red neuronal pueda procesar las imágenes, se transforman en vectores de 784 valores (28x28 píxeles) y se normalizan dividiéndolos entre 255 para que sus valores estén entre 0 y 1. Además, las etiquetas se convierten a formato **one-hot encoding**, lo que permite clasificar cada dígito de manera eficiente.
```python
x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
y_train = to_categorical(train_labels_y)
```

### 3. Definición de la Red Neuronal
El modelo está compuesto por:
- **Capa de entrada:** 784 neuronas (28x28 píxeles)
- **Capa oculta:** 512 neuronas con activación ReLU
- **Capa de salida:** 10 neuronas (una por cada dígito), con activación Softmax para obtener probabilidades
```python
model = Sequential([
    Input(shape=(28*28,)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])
```

### 4. Compilación del Modelo
Se utiliza el optimizador RMSprop, una función de pérdida **categorical crossentropy** (para clasificación multiclase) y la métrica de precisión para evaluar el desempeño del modelo.
```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 5. Entrenamiento del Modelo
El modelo se entrena durante 8 épocas con lotes de 128 imágenes a la vez.
```python
model.fit(x_train, y_train, epochs=8, batch_size=128)
```

### 6. Evaluación del Modelo
Después del entrenamiento, se prueba el modelo con las imágenes de prueba para medir su precisión.
```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")
```

## Cómo Ejecutar el Código
Para ejecutar el código, simplemente usa el siguiente comando en la terminal:
```bash
python main.py
```
El archivo `main.py` importará y ejecutará la función de entrenamiento desde `src/neural_networks_keras.py`.

## Resultados Esperados
Después de ejecutar el código, deberías ver un mensaje en la consola con la precisión obtenida en el conjunto de prueba, similar a:
```
Precisión en el conjunto de prueba: 0.97
```
Esto significa que el modelo es capaz de reconocer los dígitos con un 97% de precisión.

## Conclusión
Este proyecto demuestra cómo una red neuronal simple puede reconocer dígitos escritos a mano con alta precisión utilizando la biblioteca Keras. Se puede mejorar el modelo aumentando el número de épocas, ajustando la arquitectura o probando diferentes optimizadores.

---

¡Gracias por revisar este proyecto! Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue en el repositorio. 🚀

