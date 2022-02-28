import numpy as np

x = np.load("Anomaly_detection_slider_x.npy")
y = np.load("Anomaly_detection_slider_y.npy")

print(x.shape)
print(y.shape)

NUM_TRAINING_DATA = 1600    		# Max 1779
NUM_TEST_DATA = 1780-NUM_TRAINING_DATA

def generate_training_data():

    with open("Anomaly_detection_training_data.h", "w") as f:    
        f.write("extern const float Anomaly_detection_training_data[" + str(NUM_TRAINING_DATA) + "][" + str(x.shape[1]) + "];\n")  
        f.write("extern const float Anomaly_detection_training_data_label[" + str(NUM_TRAINING_DATA) + "];\n")

    with open("Anomaly_detection_training_data.c", "w") as f:    
        f.write("const float Anomaly_detection_training_data[" + str(NUM_TRAINING_DATA) + "][" + str(x.shape[1]) + "] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write("},\n")
            f.write("{" + str(x[i][0]) + "f")
            for j in range(1,x.shape[1]):
                f.write(", " + str(x[i][j]) + "f")
        f.write("}\n};\n")
  
        f.write("const float Anomaly_detection_training_data_label[" + str(NUM_TRAINING_DATA) + "] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write(",\n")
            f.write(str(float(y[i])) + "f")
        f.write("};")

def generate_test_data():

    with open("Anomaly_detection_test_data.h", "w") as f:    
        f.write("extern const float Anomaly_detection_test_data[" + str(NUM_TEST_DATA) + "][" + str(x.shape[1]) + "];\n")  
        f.write("extern const float Anomaly_detection_test_data_label[" + str(NUM_TEST_DATA) + "];\n")

    with open("Anomaly_detection_test_data.c", "w") as f:    
        f.write("const float Anomaly_detection_training_data[" + str(NUM_TEST_DATA) + "][" + str(x.shape[1]) + "] = {\n")
        for i in range(0,NUM_TEST_DATA):
            if i != 0:
                f.write("},\n")
            f.write("{" + str(x[i][0]) + "f")
            for j in range(1,x.shape[1]):
                f.write(", " + str(x[i][j]) + "f")
        f.write("}\n};\n")
  
        f.write("const float Anomaly_detection_test_data_label[" + str(NUM_TEST_DATA) + "] = {\n")
        for i in range(0,NUM_TEST_DATA):
            if i != 0:
                f.write(",\n")
            f.write(str(float(y[i])) + "f")
        f.write("};")

generate_training_data()
generate_test_data()