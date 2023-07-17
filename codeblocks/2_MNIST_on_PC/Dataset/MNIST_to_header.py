''' 
Author: Hahn-Schickard-Gesellschaft für angewandte Forschung e.V., Daniel Konegen + Marcus Rueb
'''

import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# Load and preprocess the MNIST data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(float)/255.0
x_test = x_test.astype(float)/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputSize = 1
for i in range(1, x_train.ndim):
    inputSize = inputSize * x_train.shape[i]

print("Number of input  neurons: " + str(inputSize))
print("Number of output neurons: " + str(y_train.shape[1]))
print("x_train" + str(x_train.shape))
print("y_train" + str(y_train.shape))
print("x_test " + str(x_test.shape))
print("y_test " + str(y_test.shape))

dataset_name = "MNIST"
NUM_TRAINING_DATA = 6000   # Max 60000
NUM_TEST_DATA = 1000       # Max 10000

def generate_training_data():

    with open(dataset_name+"_training_data.h", "w") as f:    
        f.write("extern const float "+dataset_name+"_training_data[" + str(NUM_TRAINING_DATA) + "][" + str(inputSize) + "];\n")  
        f.write("extern const float "+dataset_name+"_training_data_label[" + str(NUM_TRAINING_DATA) + "][" + str(y_train.shape[1]) + "];\n")

    with open(dataset_name+"_training_data.c", "w") as f:    
        f.write("const float "+dataset_name+"_training_data[" + str(NUM_TRAINING_DATA) + "][" + str(inputSize) + "] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write("},\n")
            x_train_flattened = x_train[i].flatten()
            f.write("{" + str(x_train_flattened[0]) + "f")
            for j in range(1,inputSize):
                f.write(", " + str(x_train_flattened[j]) + "f")
        f.write("}\n};\n")
  
        f.write("const float "+dataset_name+"_training_data_label[" + str(NUM_TRAINING_DATA) + "][" + str(y_train.shape[1]) + "] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write("},\n")
            f.write("{" + str(y_train[i][0]) + "f")
            for j in range(1,y_train.shape[1]):
                f.write(", " + str(y_train[i][j]) + "f")
        f.write("}\n};")      
        

def generate_test_data():

    with open(dataset_name+"_test_data.h", "w") as f:    
        f.write("extern const float "+dataset_name+"_test_data[" + str(NUM_TEST_DATA) + "][" + str(inputSize) + "];\n")  
        f.write("extern const float "+dataset_name+"_test_data_label[" + str(NUM_TEST_DATA) + "][" + str(y_test.shape[1]) + "];\n")

    with open(dataset_name+"_test_data.c", "w") as f:    
        f.write("const float "+dataset_name+"_test_data[" + str(NUM_TEST_DATA) + "][" + str(inputSize) + "] = {\n")
        for i in range(0, NUM_TEST_DATA):
            if i != 0:
                f.write("},\n")
            x_test_flattened = x_test[i].flatten()
            f.write("{" + str(x_test_flattened[0]) + "f")
            for j in range(1,inputSize):
                f.write(", " + str(x_test_flattened[j]) + "f")
        f.write("}\n};\n")
  
        f.write("const float "+dataset_name+"_test_data_label[" + str(NUM_TEST_DATA) + "][" + str(y_test.shape[1]) + "] = {\n")
        for i in range(0, NUM_TEST_DATA):
            if i != 0:
                f.write("},\n")
            f.write("{" + str(y_test[i][0]) + "f")
            for j in range(1,y_test.shape[1]):
                f.write(", " + str(y_test[i][j]) + "f")
        f.write("}\n};")


generate_training_data()
generate_test_data()