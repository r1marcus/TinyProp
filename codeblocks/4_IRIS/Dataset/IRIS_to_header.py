''' 
Author: Hahn-Schickard-Gesellschaft für angewandte Forschung e.V., Daniel Maier
'''

import numpy
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset_name = "IRIS"
NUM_TRAINING_DATA = 120    		# Max 150
NUM_TEST_DATA = 150-NUM_TRAINING_DATA

# Load and preprocess the dataset
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, to_categorical(iris.target), 
                    test_size=NUM_TEST_DATA, train_size=NUM_TRAINING_DATA, random_state=42)

def generate_training_data():

    with open(dataset_name+"_training_data.h", "w") as f:    
        f.write("extern const float "+dataset_name+"_training_data[" + str(NUM_TRAINING_DATA) + "][" + str(x_train.shape[1]) + "];\n")  
        f.write("extern const float "+dataset_name+"_training_data_label[" + str(NUM_TRAINING_DATA) + "][" + str(y_train.shape[1]) + "];\n")

    with open(dataset_name+"_training_data.c", "w") as f:    
        f.write("const float "+dataset_name+"_training_data[" + str(NUM_TRAINING_DATA) + "][" + str(x_train.shape[1]) + "] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write("},\n")
            f.write("{" + str(x_train[i][0]) + "f")
            for j in range(1,x_train.shape[1]):
                f.write(", " + str(x_train[i][j]) + "f")
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
        f.write("extern const float "+dataset_name+"_test_data[" + str(NUM_TEST_DATA) + "][" + str(x_test.shape[1]) + "];\n")  
        f.write("extern const float "+dataset_name+"_test_data_label[" + str(NUM_TEST_DATA) + "][" + str(y_test.shape[1]) + "];\n")

    with open(dataset_name+"_test_data.c", "w") as f:    
        f.write("const float "+dataset_name+"_test_data[" + str(NUM_TEST_DATA) + "][" + str(x_test.shape[1]) + "] = {\n")
        for i in range(0, NUM_TEST_DATA):
            if i != 0:
                f.write("},\n")
            f.write("{" + str(x_test[i][0]) + "f")
            for j in range(1,x_test.shape[1]):
                f.write(", " + str(x_test[i][j]) + "f")
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