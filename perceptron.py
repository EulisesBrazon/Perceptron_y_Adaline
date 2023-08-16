import csv  # manejo de un tipo de archivo
import os  # para el manejo de rutas del sistema

import matplotlib.pyplot as plt  # graficar datos
import numpy as np  # libreria para la manipulacion de matrices y arreglos de forma eficiente

class Perceptron:
    def __init__(self, num_inputs):

        # se inicializan una semilla, con pesos aleatoreos,
        # la cantidad de pesos depende de la cantidad de entradas
        self.weights = np.random.randn(num_inputs)

        # bias es un parametro que permite ajustar el umbral de activacion
        # valor a partir del cual la funcion de activacion se activa o no 
        # vinculado con la influencia que tiene x0
        self.bias = 0.0

        self.num_features = num_inputs

    def sign_funtion(self, presynaptic_potential):
        function = 1 if presynaptic_potential >= 0 else -1
        return function
    
    def scalar_product(self, inputs):
        scalar = np.dot(inputs, self.weights) + self.bias
        return scalar

    def predict(self, inputs):
        # inputs representa el vector de datos de entrada
        # la suma total, de los pesos multiplicado por las entradas, se traduce como un producto escalar
        # entre el vector de entrada y el vector de pesos, con ayuda de np.dot(), se optiene el producto escalar
        # y el resultado de la suma ponderada total, se almacena en weighted_sum
        weighted_sum = self.scalar_product(inputs)
        # una vez obtenido un valor, es pasado por una funcion de activacion, que en este caso corresponde con 
        # la funcion signo  , que retorna 1 o -1 dependiendo si el resultado es menor o mayor igual que 0 
        activation = self.sign_funtion(weighted_sum)
        return activation
    
    #Error cuadratico medio
    def quadratic_error(self, label, prediction): 
        error = ((label - prediction) ** 2)/2 
        return error
    
    def train(self, training_inputs, labels, show_data= True, num_epochs=1000, learning_rate = 0.01):
        count = 0
        accumulate_error = 1
        # se crea un recorido para el numero de epocas, 
        # veces que seran procesado el conjunto de datos 
        # de entrenamiento para ajustar los pesos
        while (count<num_epochs) and (accumulate_error > 0): 

            accumulate_error = 0

            # el conjunto de entrenamiento se encuentra dividido en la entrada de dados,
            #  y la salida esperada para esa entrada de datos ej: [(x1,x2)(salida)]
            for inputs, label in zip(training_inputs, labels):

                #aqui se aplica la funcion escalonada al producto scalar de las entrada por los pesos
                prediction = self.predict(inputs) 

                # error producido en esta evaluacion
                error = label - prediction

                # error acumulado de las multiples iteraciones
                accumulate_error += self.quadratic_error(label, prediction)

                # se modifican el valor de los pesos para cada una de las entradas
                self.weights += [learning_rate * error * input_val for input_val in inputs]

                # se modifica el peso asociado a la entrada x0
                self.bias += learning_rate * error

            count += 1

        if show_data:
            print(f"Iteraciones sobre el conjunto de entrenamiento entrenamiento del Perceptron: {count}")
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
        y_max = ((m*x_max) + b)
        y_min = ((m*x_min) + b)


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

                # carga todos los valores exeptuando la ultima columna
                training_inputs.append([float(value) for value in row[:-1]])

                # carga la ultima comulna
                labels.append(int(row[-1]))

        return training_inputs, labels
    
     # graficar las basado en coordenadas bidimencional        
    def graph(self, training_inputs, labels, x_ask = None, y_ask = None):
        x1 = []
        x2 = []

        # recorido para separar las coordenadas
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
        plt.title('Hiperplano separador Perceptron')

        # Mostrar el gráfico
        plt.show()



def main(training_data):
    # en este caso el perceptron contara con dos entrada x1,x2, x0 esta considerado dentro del Perceptron con el bias
    program = Perceptron(2)

    # se cargan los datos de entrenamiento
    training_inputs, labels = program.load_data_csv(training_data)

    # se procede a realizar el entrenamiento para ajustar los pesos
    program.train(training_inputs=training_inputs, labels=labels)

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
