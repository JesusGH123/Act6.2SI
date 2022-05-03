#Código obtenido de: https://www.allaboutcircuits.com/technical-articles/how-to-create-a-multilayer-perceptron-neural-network-in-python/
import pandas
import numpy as np
import openpyxl

#Función de activación
def logistic(x):
    return 1.0/(1 + np.exp(-x))

#gradiente descendiente
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

LR = 0.25 #Tasa de aprendizaje
error_aceptable = 0.02 #Este error se puede utilizar en vez de las epocas para terminar el programa
FE = 0 #Factor de error que durante las iteraciones se irá actualizando
I_dim = 2 #Numero de neuronas en capa de entrada
H_dim = 2 #Numero de neuronas en la capa escondida

epoch_count = 1000000 #Numero de epocas

#np.random.seed(1)
weights_ItoH = np.asanyarray([[0.1,0.5],[-0.7,0.3]]) #np.random.uniform(-1,1,(I_dim,H_dim))
weights_HtoO = np.asanyarray([0.2,0.4]) #np.random.uniform(-1,1,H_dim)

preActivation_H = np.zeros(H_dim) #Arreglo que almacenará los productos punto de los nodos de entrada
postActivation_H = np.zeros(H_dim) #Arreglo que almacenará el que retorna la función de activación

#Se sube un documento de excel con los valores de entrenamiento que tenga 3 columnas x1, x2 y output real
training_data = pandas.read_excel('training.xlsx')
target_output = training_data.output
training_data = training_data.drop(['output'], axis=1)
training_data = np.asarray(training_data)
training_count = len(training_data[:,0])

#Se sube un documento de excel con los valores de entrenamiento que tenga 3 columnas x1, x2 y output real
validation_data = pandas.read_excel('validation.xlsx')
validation_output = validation_data.output
validation_data = validation_data.drop(['output'], axis=1)
validation_data = np.asarray(validation_data)
validation_count = len(validation_data[:,0])

#El algoritmo iterará de acuerdo a las epocas que le proporcionemos
for epoch in range(epoch_count):
    for sample in range(training_count):
        for node in range(H_dim):
            #Calculamos la el producto punto de los inputs y pesos y la función de activación para cada neurona que sea input
            preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
            postActivation_H[node] = logistic(preActivation_H[node])

        #Calculamos la el producto punto de las neuronas escondidas y sus pesos y la función de activación para cada neurona que sea input
        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)

        #Calculamos el factor de error. El resultado de la neurona - el valor real
        FE = postActivation_O - target_output[sample]

        #A partir de aqui empieza la retro-propagacion desde la capa oculta a la input
        for H_node in range(H_dim):
            #Calculamos el delta k para empezar a actualizar los pesos de la capa oculta
            S_error = FE * logistic_deriv(preActivation_O)
            gradient_HtoO = S_error * postActivation_H[H_node]

            #Calculamos el delta k para empezar a actualizar los pesos de la capa input
            for I_node in range(I_dim):
                input_value = training_data[sample, I_node]
                gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value
                #Aplicamos la formula para actualizar los pesos de la capa input.
                #Formula Peso_nuevo = Peso_actual + tasa de aprendizaje * valor de la neurona de entrada* delta k
                weights_ItoH[I_node, H_node] -= LR * gradient_ItoH
            #Aplicamos la formula para actualizar los pesos de la capa output.
                #Formula Peso_nuevo = Peso_actual + tasa de aprendizaje * valor de la neurona de salida * delta k
            weights_HtoO[H_node] -= LR * gradient_HtoO


#En esta parte hacemos la propagacion hacia adelante para probar los datos de validacion, esto nos tiene que dar un valor aproximado al real 
correct_classification_count = 0
for sample in range(validation_count):
    for node in range(H_dim):
        preActivation_H[node] = np.dot(validation_data[sample,:], weights_ItoH[:, node])
        postActivation_H[node] = logistic(preActivation_H[node])
            
    preActivation_O = np.dot(postActivation_H, weights_HtoO)
    postActivation_O = logistic(preActivation_O)

    #Mostramos el valor de salida de cada par de elementos
    print("Postactivation: of ", training_data[sample,:], ": ", postActivation_O)

    #Si el valor es mayor a 55 damos por hecho de que será 1, si no, es 0. Aunque el valor real se imprime en la linea de arriba
    if postActivation_O > 0.55:
        output = 1
    else:
        output = 0     
        
    if output == validation_output[sample]:
        correct_classification_count += 1

print("Pesos finales de la capa de entrada: ")
for weight_i in weights_ItoH:
  print(weight_i)
print("Pesos finales de la capa escondida: ")
for weight_h in weights_HtoO:
  print(weight_h)
  
print('Percentage of correct classifications:')
print(correct_classification_count*100/validation_count)