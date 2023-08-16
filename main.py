import csv  # manejo de un tipo de archivo
import os  # para el manejo de rutas del sistema

import matplotlib.pyplot as plt  # graficar datos
import numpy as np  # libreria para la manipulacion de matrices y arreglos de forma eficiente
from perceptron import Perceptron
from adaline import Adaline


def generate_recta(decimals = 2, min = 0, max = 10):
    # Generar puntos aleatorios dentro del rango [min, max]
    x1, y1 = round(np.random.uniform(min, max), decimals), round(np.random.uniform(min, max), decimals)
    x2, y2 = round(np.random.uniform(min, max), decimals), round(np.random.uniform(min, max), decimals)

    # Calcular la pendiente 'm' de la recta
    m = (y2 - y1) / (x2 - x1)

    # Escoger un valor aleatorio para 'b'
    b = round(np.random.uniform(min, max), decimals)

    print(f"La recta generada es y = {m:.2f}x + {b:.2f}.")
    return m, b

def generate_training_data_csv(file_name, num_samples=100, decimals = 2, min = 0, max = 10):

    #generar recta Aleatorea
    m, b = generate_recta()

    #Obtener la ruta absoluta del archivo principal (main)
    ruta_principal = os.path.abspath(__file__)

    # Construir la ruta del archivo de salida
    ruta = os.path.join(os.path.dirname(ruta_principal), file_name)

    # Crear archivo CSV de entrenamiento con las coordenadas y las etiquetas
    with open(ruta, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['x1', 'x2', 's'])

        for _ in range(num_samples):
            #generar numeros aleatoreos
            x = round(np.random.uniform(0, 10), decimals)
            y = round(np.random.uniform(0, 10), decimals)

            # Clasificar el punto según la recta generada
            label = 1 if y >= m*x + b else -1

            writer.writerow([x, y, label])

    print(f"Se ha generado el archivo {file_name} con {num_samples} puntos de entrenamiento.")

def load_data_csv(file_name):
        training_inputs = []
        labels = []

        # Obtener la ruta absoluta del archivo principal (main)
        ruta_principal = os.path.abspath(__file__)

        # Construir la ruta del archivo de salida
        ruta = os.path.join(os.path.dirname(ruta_principal), file_name)

        #se abre el archivo en modo lectura
        with open(ruta, 'r') as archivo:
            reader_csv = csv.reader(archivo)
            for row in reader_csv:

                # carga todos los valores exceptuando  la ultima columna
                training_inputs.append([float(value) for value in row[:-1]])

                # carga la ultima comulna
                labels.append(int(row[-1]))

        return training_inputs, labels

def main(training_data, create_data=True):

    # se generan los datos de entrenamiento y se almacenan en un .csv
    if create_data:
        generate_training_data_csv(training_data)

    # se cargan los datos de entrenamiento
    training_inputs, labels = load_data_csv(training_data)

    #tipo de dato necesario para hacer uso de la libreria np en la realizacion de los calculos
    training_inputs=np.array(training_inputs)
    labels=np.array(labels)

    perceptron = Perceptron(num_inputs=2)
    adaline = Adaline(num_inputs=2) 


    # se procede a realizar el entrenamiento para ajustar los pesos
    perceptron.train(training_inputs, labels)
    adaline.train(training_inputs, labels)

    #graficar Resultado del perceptron
    perceptron.graph(training_inputs, labels)
    adaline.graph(training_inputs, labels)

    print(f"\nRed entranada con {training_data}")

if __name__ == "__main__":
    try:
        training_data = "training_data.csv"
        main(training_data)
        
    # en caso de error muestra los detalles del error
    except Exception as e:
        print("Error:", e)



