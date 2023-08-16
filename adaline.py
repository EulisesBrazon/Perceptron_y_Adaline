import csv  # manejo de un tipo de archivo
import os  # para el manejo de rutas del sistema

import matplotlib.pyplot as plt  # graficar datos
import numpy as np  # libreria para la manipulacion de matrices y arreglos de forma eficiente

class Adaline:

    def __init__(self, learning_rate=0.01, num_epochs=1000, num_inputs=2):
        self.eta = learning_rate  # Taza de aprendizaje
        self.num_epochs = num_epochs  # Numero de iteraciones
        self.weights = np.random.randn(num_inputs) 
        self.num_features = num_inputs
        self.bias = 0.0

    def net_input(self, X):
        #Producto escalar
        return np.dot(X, self.weights) + self.bias 

    def activation(self, X):
        #funcion signo
        return np.where(X >= 0.0, 1, -1)

    def predict(self, X):
        #potencial precinaptico pasado por la funcion de activacion
        return self.activation(self.net_input(X))

    def train(self, training_inputs, labels, show_data = True):

        for i in range(self.num_epochs):
            #se obtiene el producto escalar
            net_input = self.net_input(training_inputs)
            #el resultado de pasa por la funcion de activacion
            output = self.activation(net_input)
            #se realiza una resta entre dos matrices
            errors = (labels - output)

            #se actualizan los pesos aplicando deceso de gradiente que se traduce en
            self.weights += self.eta * training_inputs.T.dot(errors)
            self.bias += self.eta * errors.sum()  
        if show_data:
            print("pesos ajustados en el Adaline")
            print(f"Peso w0: {self.bias }")
            for i in range(self.num_features):
                print(f"Peso w{i+1}: {self.weights[i] }") 
   
    def coord_hyperplane(self, training_inputs):
        # hiperplano separador
        # de la formula 
        # w1x1 + w2x2 + w0 = 0
        # despajando x2 obtenemos
        # x2 = -(w1/w2)x1 - w0/w2
        # siendo que y = mx + b
        # pudidendo representar m como -(w1/w2)
        # y b como -w0/w2
        # se hacen los resplazos correspondiente para evaluar
        # las posiciones en 0 y se optiene

        m = -1*(self.weights[0]/self.weights[1])
        b = -1*(self.bias/self.weights[1])

        # cuando x=0 y es igual a
        # y0 = -1*(self.bias / self.weights[1])
 
        # se determina la coordenada en x de los dos extremos
        x_max = max(row[0]for row in training_inputs)
        x_min = min(row[0]for row in training_inputs)

        # se determina con coodenada en y correspondientes a las x mas lejanas
        y_max = (m*x_max) + b
        y_min = (m*x_min) + b

        return [x_min,x_max],[y_min,y_max]

    def load_data_csv(self, file_name):
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
    
     # graficar las basado en coordenadas bidimensional                
    def graph(self, training_inputs, labels, x_ask = None, y_ask = None):
        x1 = []
        x2 = []

        # recorrido  para separar las coordenadas
        for x1_train,x2_train in training_inputs:
            x1.append(x1_train)
            x2.append(x2_train)

        # Obtener colores correspondientes depoendiendo de la salida
        colors = ['red' if value == 1 else 'blue' for value in labels]

        # agregando coordenada de prengunta
        x1.append(x_ask)
        x2.append(y_ask)
        colors.append("green")

        # Graficar los puntos con colores
        plt.scatter(x1, x2, c=colors)

        # generar dos puntos de coordenadas para grafiar la recta del hiperplano
        x_coords,y_coords = self.coord_hyperplane(training_inputs)

        # Graficar la recta
        plt.plot(x_coords, y_coords, color="orange")
        
        # Configuraciones adicionales
        plt.xlabel('Entrada x/w1')
        plt.ylabel('Entrada y/w2')
        plt.title('Hiperplano separador Adaline')

        # Mostrar el gráfico
        plt.show()

def main(training_data):
    # en este caso el perceptron contara con dos entrada x1,x2, x0 esta considerado  dentro del Perceptron con el bias
    program = Adaline(num_inputs=2)

    # se cargan los datos de entrenamiento
    training_inputs, labels = program.load_data_csv(training_data)

    # se procede a realizar el entrenamiento para ajustar los pesos
    program.train(training_inputs=np.array(training_inputs), labels=np.array(labels))

    print(f"\nRed entranada con {training_data}")

    while True:
        x_ask = input("Ingrese un coordenada x/w1: ")

        # La cadena contiene un número entero positivo, negativo o decimal válido
        if x_ask.lstrip('-').replace('.', '', 1).isdigit():
            x_ask = int(x_ask)
            print("Número válido x/w1:", x_ask)
            break
        else:
            print("Entrada inválida. Por favor, ingrese solo números")

    while True:
        y_ask = input("Ingrese un coordenada y/w2: ")

        # La cadena contiene un número entero positivo, negativo o decimal válido
        if y_ask.lstrip('-').replace('.', '', 1).isdigit():
            y_ask = int(y_ask)
            print("Número válido y/w2:", y_ask)
            break
        else:
            print("Entrada inválida. Por favor, ingrese solo números")

    result = program.predict([x_ask,y_ask])
    result = ['Rojo' if result == 1 else 'Azul']
    
    print("\nEl punto (", x_ask,",", y_ask,") pertenece al grupo :", result[0])
    program.graph(training_inputs, labels, x_ask, y_ask)

if __name__ == "__main__":
    try:
        training_data = "training_data.csv"
        main(training_data)
        
    # en caso de error muestra los detalles del error
    except Exception as e:
        print("Error:", e)
