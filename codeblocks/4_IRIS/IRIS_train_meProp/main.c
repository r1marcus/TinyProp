/*
  Author: Hahn-Schickard-Gesellschaft für angewandte Forschung e.V., Daniel Maier

  The sketch shows an example of how a neural network is trained from scratch in AIfES using training data.
  This is a script to train a neural network for the IRIS dataset on a PC.

  If you need help to execute the example look here:
  https://www.hackster.io/aifes_team/how-to-use-aifes-on-a-pc-or-in-other-ides-ef20a0
*/

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "aifes.h"
#include <time.h>

#include "IRIS_training_data.h"
#include "IRIS_test_data.h"

#define NUM_DENSE_LAYERS 3
#define INPUTS  4
#define NEURONS_1 8
#define NEURONS_2 6
#define OUTPUTS 3
#define NUM_TRAIN_DATA 120
#define NUM_TEST_DATA  30
#define EPOCHS 100
#define BATCH_SIZE 10

#define MIN_BPR 0.1f
#define MAX_BPR 0.9f
#define DAMPING 0.5f


void writeCsvFile(char *filename, float data[][EPOCHS+1][NUM_DENSE_LAYERS+5], unsigned int *seeds, int trials){
    FILE *fp;
    int i,j, n=NUM_DENSE_LAYERS+5, m=EPOCHS+1;

    filename=strcat(filename,".csv");
    fp=fopen(filename,"w+");

    //create header
    fprintf(fp,"Epoch,Training Loss,Training Accuracy,Time [s]");
    for (int i = 0; i<NUM_DENSE_LAYERS; i++) {
        fprintf(fp,",BPR Layer %i",i+1);
    }
    fprintf(fp,",Test Loss,Test Accuracy,Seed");

    //write data
    for (int t=0; t<trials; t++) {
        fprintf(fp,"\n%i,%.3f,%.3f",0,data[t][0][0],data[t][0][1]);
        for(i=1;i<m;i++){
            fprintf(fp,"\n%d",i);
            for(j=0;j<n;j++)
                fprintf(fp,",%.3f",data[t][i][j]);
        }
        // print additional info
        fprintf(fp,",%i\n",seeds[t]);
    }
    fclose(fp);
}


int main(int argc, char *argv[])
{
    char str[24];
    printf("\nEnter filename for documentation: ");
    gets(str);
    int trials;
    printf("\nEnter number of trials: ");
    scanf ("%d",&trials);


    uint16_t input_shape[] = {NUM_TRAIN_DATA, INPUTS};              // Definition of the input shape
    aitensor_t input_tensor;                                        // Creation of the input AIfES tensor
      input_tensor.dtype = aif32;                                   // Definition of the used data type, here float with 32 bits, different ones are available
      input_tensor.dim = 2;                                         // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      input_tensor.shape = input_shape;                             // Set the shape of the input_tensor
      input_tensor.data = IRIS_training_data;                      // Assign the input_data array to the tensor. It expects a pointer to the array where the data is stored


    uint16_t input_shape_test_data[] = {NUM_TEST_DATA, INPUTS};     // Definition of the input shape
    aitensor_t input_tensor_test_data;                              // Creation of the input AIfES tensor
      input_tensor_test_data.dtype = aif32;                         // Definition of the used data type, here float with 32 bits, different ones are available
      input_tensor_test_data.dim = 2;                               // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      input_tensor_test_data.shape = input_shape_test_data;         // Set the shape of the input_tensor
      input_tensor_test_data.data = IRIS_test_data;                // Assign the input_data array to the tensor. It expects a pointer to the array where the data is stored


    uint16_t target_shape[] = {NUM_TRAIN_DATA, OUTPUTS};            // Definition of the output shape
    aitensor_t target_tensor;                                       // Creation of the input AIfES tensor
      target_tensor.dtype = aif32;                                  // Definition of the used data type, here float with 32 bits, different ones are available
      target_tensor.dim = 2;                                        // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      target_tensor.shape = target_shape;                           // Set the shape of the input_tensor
      target_tensor.data = IRIS_training_data_label;               // Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored


    uint16_t target_shape_test_data[] = {NUM_TEST_DATA, OUTPUTS};   // Definition of the output shape
    aitensor_t target_tensor_test_data;                             // Creation of the input AIfES tensor
      target_tensor_test_data.dtype = aif32;                        // Definition of the used data type, here float with 32 bits, different ones are available
      target_tensor_test_data.dim = 2;                              // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      target_tensor_test_data.shape = target_shape_test_data;       // Set the shape of the input_tensor
      target_tensor_test_data.data = IRIS_test_data_label;         // Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored

    // Tensor for the output data (result after training).
    // Same configuration as for the target tensor
    float output_data[NUM_TRAIN_DATA][OUTPUTS];
    uint16_t output_shape[] = {NUM_TRAIN_DATA, OUTPUTS};
    aitensor_t output_tensor;
      output_tensor.dtype = aif32;
      output_tensor.dim = 2;
      output_tensor.shape = output_shape;
      output_tensor.data = output_data;

	// Tensor for the output test data (result of testing).
    // Same configuration as for the target tensor
    float output_test_data[NUM_TEST_DATA][OUTPUTS];
    uint16_t output_shape_test_data[] = {NUM_TEST_DATA, OUTPUTS};
    aitensor_t output_tensor_test_data;
      output_tensor_test_data.dtype = aif32;
      output_tensor_test_data.dim = 2;
      output_tensor_test_data.shape = output_shape_test_data;
      output_tensor_test_data.data = output_test_data;

    // ---------------------------------- Layer definition ---------------------------------------
    ailayer_dense_t dense_layers[NUM_DENSE_LAYERS];                 // Creation of the AIfES hidden dense layers


    // Input layer
    uint16_t input_layer_shape[] = {1, INPUTS};                     // Definition of the input layer shape (Must fit to the input tensor)
    ailayer_input_t input_layer;                                    // Creation of the AIfES input layer
    input_layer.input_dim = 2;                                      // Definition of the input dimension (Must fit to the input tensor)
    input_layer.input_shape = input_layer_shape;                    // Handover of the input layer shape

    // Dense layer (hidden layer)
    dense_layers[0].neurons = NEURONS_1;                            // Number of neurons
    ailayer_relu_t relu_layer_1;                                    // Relu activation function


    // Dense layer (hidden layer_2)
    dense_layers[1].neurons = NEURONS_2;                            // Number of neurons
    ailayer_relu_t relu_layer_2;                                    // Relu activation function

    // Output dense layer
    dense_layers[2].neurons = OUTPUTS;                              // Number of neurons
    ailayer_softmax_t softmax_layer_3;                              // Softmax activation function

    ailoss_crossentropy_t crossentropy_loss;                        // Loss: Crossentropy

    // --------------------------- Define the structure of the model ----------------------------

    aimodel_t model;                                                // AIfES model
    ailayer_t *x;                                                   // Layer object from AIfES, contains the layers

    // Passing the layers to the AIfES model
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layers[0], model.input_layer);
    x = ailayer_relu_f32_default(&relu_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layers[1], x);
    x = ailayer_relu_f32_default(&relu_layer_2, x);
    x = ailayer_dense_f32_default(&dense_layers[2], x);
    x = ailayer_softmax_f32_default(&softmax_layer_3, x);
    model.output_layer = x;

    // Add the loss to the AIfES model
    model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);

    aialgo_compile_model(&model); // Compile the AIfES model

    // ------------------------------- Allocate memory for the parameters of the model ------------------------------
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    printf("Required memory for parameter (Weights, Bias, ...):");
    printf("%d",parameter_memory_size);
    printf("Byte\n");

    void *parameter_memory = malloc(parameter_memory_size);

    // Distribute the memory to the trainable parameters of the model
    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    // -------------------------------- Define the optimizer for training ---------------------

    aiopti_t *optimizer; // Object for the optimizer

    //ADAM optimizer
    aiopti_adam_f32_t adam_opti;
    adam_opti.learning_rate = 0.001f;
    adam_opti.beta1 = 0.9f;
    adam_opti.beta2 = 0.999f;
    adam_opti.eps = 1e-7;

    // Choose the optimizer
    optimizer = aiopti_adam_f32_default(&adam_opti);

    // -------------------------------- Allocate and schedule the working memory for training ---------

    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    printf("Required memory for the training (Intermediate results, gradients, optimization memory): %d Byte\n", memory_size);

    void *memory_ptr = malloc(memory_size);

    // Schedule the memory over the model
    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

    // Initialize the AIfES model
    aialgo_init_model_for_training(&model, optimizer);

    // documentation
    float history[trials][EPOCHS+1][NUM_DENSE_LAYERS+5];  //before training + epochs
    unsigned int seeds[trials];

    for (int t=0; t<trials; t++) {
        printf("\nStarting trial %i...\n",t+1);
        // ------------------------------- Initialize the parameters ------------------------------

        seeds[t] = time(NULL);
        //seeds[t] = 1634302329;
        srand(seeds[t]);

        aimath_f32_default_init_glorot_uniform(&dense_layers[0].weights);
        aimath_f32_default_init_zeros(&dense_layers[0].bias);
        dense_layers[0].maxError = 0;
        aimath_f32_default_init_glorot_uniform(&dense_layers[1].weights);
        aimath_f32_default_init_zeros(&dense_layers[1].bias);
        dense_layers[1].maxError = 0;
        aimath_f32_default_init_glorot_uniform(&dense_layers[2].weights);
        aimath_f32_default_init_zeros(&dense_layers[2].bias);
        dense_layers[2].maxError = 0;

        // ------------------------------------- Run the training ------------------------------------

        for(uint32_t i = 0; i < EPOCHS; i++)
        {
            float loss, acc;
            for (int l = 0; l<NUM_DENSE_LAYERS; l++) {
                dense_layers[l].ratio = 0.0f;
            }

            aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
            aialgo_inference_model(&model, &input_tensor, &output_tensor);
            predict_model_acc(&acc, NUM_TRAIN_DATA, OUTPUTS, &IRIS_training_data_label, &output_data);
            if (i > 0) {
                float loss_test, test_acc;
                aialgo_inference_model(&model, &input_tensor_test_data, &output_tensor_test_data);
                predict_model_acc(&test_acc, NUM_TEST_DATA, OUTPUTS, &IRIS_test_data_label, &output_test_data);
                aialgo_calc_loss_model_f32(&model, &input_tensor_test_data, &target_tensor_test_data, &loss_test);
                //reduction / normalization of the loss
                loss_test = loss_test / (OUTPUTS * NUM_TEST_DATA);
                history[t][i][NUM_DENSE_LAYERS+3] = loss_test;
                history[t][i][NUM_DENSE_LAYERS+4] = test_acc;
                printf("\rEpoch %d completed, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%", i, acc, test_acc);
            }
            else {
                printf("\rAccuracy before Training: %.2f%%", acc);

            }

            history[t][i][0] = loss/(OUTPUTS * NUM_TRAIN_DATA);
            history[t][i][1] = acc;

            // One epoch of training. Iterates through the whole data once
			clock_t start = clock();
            aialgo_train_model_meProp(&model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE, MAX_BPR, MIN_BPR, DAMPING);
			history[t][i+1][2] = (float)(clock() - start)/CLOCKS_PER_SEC;
			for (int l = 0; l<NUM_DENSE_LAYERS; l++) {
                history[t][i+1][l+3] = dense_layers[l].ratio/NUM_TRAIN_DATA;
            }
            // Or use this command instead to measure backwards pass performance
            //measure_backwards_pass_performance(&model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE, MAX_BPR, MIN_BPR, DAMPING, 1000);

        }
        // training complete
        float loss, acc;
        aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
        aialgo_inference_model(&model, &input_tensor, &output_tensor);
        predict_model_acc(&acc, NUM_TRAIN_DATA, OUTPUTS, &IRIS_training_data_label, &output_data);
        history[t][EPOCHS][0] = loss/(OUTPUTS * NUM_TRAIN_DATA);
        history[t][EPOCHS][1] = acc;

        float loss_test, test_acc;
        aialgo_inference_model(&model, &input_tensor_test_data, &output_tensor_test_data);
        predict_model_acc(&test_acc, NUM_TEST_DATA, OUTPUTS, &IRIS_test_data_label, &output_test_data);
        aialgo_calc_loss_model_f32(&model, &input_tensor_test_data, &target_tensor_test_data, &loss_test);
        history[t][EPOCHS][NUM_DENSE_LAYERS+3] = loss_test/(OUTPUTS * NUM_TEST_DATA);
        history[t][EPOCHS][NUM_DENSE_LAYERS+4] = test_acc;

        printf("\rEpoch %d completed, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%\nFinished training\n\n", EPOCHS, acc, test_acc);
        printf("Trial %i completed\n",t+1);
    }


    writeCsvFile(str,history,seeds,trials);
    free(parameter_memory);
    free(memory_ptr);
	return 0;
}
