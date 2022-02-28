/*
  Author: Hahn-Schickard-Gesellschaft für angewandte Forschung e.V., Daniel Maier

  The sketch shows an example of how a neural network is trained from scratch in AIfES using training data.
  This is a script to train a neural network for AnomalyDetection dataset on a PC.

  If you need help to execute the example look here:
  https://create.arduino.cc/projecthub/aifes_team/how-to-use-aifes-on-a-pc-for-training-9ad5f8?ref=user&ref_id=1924948&offset=1
*/

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "aifes.h"
#include <time.h>

#include "Anomaly_detection_training_data.h"
#include "Anomaly_detection_test_data.h"

#define INPUTS  2607
#define NEURONS_1 256
#define NEURONS_2 128
#define NEURONS_3 64
#define NEURONS_4 32
#define OUTPUTS 1
#define NUM_TRAIN_DATA 1600
#define NUM_TEST_DATA 180
#define EPOCHS 50

#define MIN_BPR 0.1f
#define MAX_BPR 0.1f
#define DAMPING 0.5f

#define BACKWARDS_PASS_REPEATS 100
//#define MONITOR_BP

void measure_backwards_pass_performance(aimodel_t *model, aitensor_t *input_tensor, aitensor_t *target_tensor, aiopti_t *optimizer, uint32_t batch_size)
{
	uint32_t i;
	float elapsedTime = 0.0f;

	aitensor_t input_batch;
	uint16_t input_batch_shape[input_tensor->dim];
	input_batch.dtype = input_tensor->dtype;
	input_batch.dim = input_tensor->dim;
	input_batch.shape = input_batch_shape;
	input_batch.tensor_params = input_tensor->tensor_params;
	aitensor_t target_batch;
	uint16_t target_batch_shape[target_tensor->dim];
	target_batch.dtype = target_tensor->dtype;
	target_batch.dim = target_tensor->dim;
	target_batch.shape = target_batch_shape;
	target_batch.tensor_params = target_tensor->tensor_params;

	uint32_t input_multiplier = 1;
	for(i = input_tensor->dim - 1; i > 0; i--)
	{
		input_multiplier *= input_tensor->shape[i];
		input_batch_shape[i] = input_tensor->shape[i];
	}
	input_multiplier *= input_tensor->dtype->size;
	input_batch_shape[0] = 1;
	uint32_t target_multiplier = 1;
	for(i = target_tensor->dim - 1; i > 0; i--)
	{
		target_multiplier *= target_tensor->shape[i];
		target_batch_shape[i] = target_tensor->shape[i];
	}
	target_multiplier *= target_tensor->dtype->size;
	target_batch_shape[0] = 1;

	uint32_t batch;

	int count = 0;
	for(batch = 0; batch < 1; batch++)
	{
		aialgo_zero_gradients_model(model, optimizer);
		for(i = 0; i < batch_size; i++)
		{
		    count++;
			input_batch.data = input_tensor->data + batch * input_multiplier * batch_size + i * input_multiplier;
			target_batch.data = target_tensor->data + batch * target_multiplier * batch_size + i * target_multiplier;

			//printf("Input batch [%d, %d, %d, %d]:\n", input_batch_shape[0], input_batch_shape[1], input_batch_shape[2], input_batch_shape[3]);
			//print_aitensor(&input_batch);

			aialgo_forward_model(model, &input_batch);
			clock_t start = clock();
			for (int i=0; i<BACKWARDS_PASS_REPEATS; i++) {
                aialgo_backward_model_meProp(model, &target_batch, MAX_BPR, MIN_BPR, DAMPING);
			}
			elapsedTime += (float)(clock() - start)/CLOCKS_PER_SEC;
		}
		aialgo_update_params_model(model, optimizer);
	}
	printf("backwards pass done %i times\n",BACKWARDS_PASS_REPEATS*count);
	printf("elapsed time: %f\n",elapsedTime);
	printf("resulting in %f us/bp\n\n",elapsedTime*1000000/BACKWARDS_PASS_REPEATS/count);
	return;
}

void create_history_data(char *filename, float data[][EPOCHS+1][10], unsigned int *seeds, int trials){
    FILE *fp;
    int i,j, n=10, m=EPOCHS+1;

    filename=strcat(filename,".csv");
    fp=fopen(filename,"w+");

    fprintf(fp,"Epoch,Training Loss,Training Accuracy,Time [s],BPR Layer1,BPR Layer2,BPR Layer3,BPR Layer4,BPR Layer5,Test Loss,Test Accuracy,Seed");

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
      input_tensor.data = Anomaly_detection_training_data;          // Assign the input_data array to the tensor. It expects a pointer to the array where the data is stored

    uint16_t input_shape_test_data[] = {NUM_TEST_DATA, INPUTS};     // Definition of the input shape
    aitensor_t input_tensor_test_data;                              // Creation of the input AIfES tensor
      input_tensor_test_data.dtype = aif32;                         // Definition of the used data type, here float with 32 bits, different ones are available
      input_tensor_test_data.dim = 2;                               // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      input_tensor_test_data.shape = input_shape_test_data;         // Set the shape of the input_tensor
      input_tensor_test_data.data = Anomaly_detection_test_data;    // Assign the input_data array to the tensor. It expects a pointer to the array where the data is stored

    uint16_t target_shape[] = {NUM_TRAIN_DATA, OUTPUTS};            // Definition of the output shape
    aitensor_t target_tensor;                                       // Creation of the input AIfES tensor
      target_tensor.dtype = aif32;                                  // Definition of the used data type, here float with 32 bits, different ones are available
      target_tensor.dim = 2;                                        // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      target_tensor.shape = target_shape;                           // Set the shape of the input_tensor
      target_tensor.data = Anomaly_detection_training_data_label;   // Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored

    uint16_t target_shape_test_data[] = {NUM_TEST_DATA, OUTPUTS};   // Definition of the output shape
    aitensor_t target_tensor_test_data;                             // Creation of the input AIfES tensor
      target_tensor_test_data.dtype = aif32;                        // Definition of the used data type, here float with 32 bits, different ones are available
      target_tensor_test_data.dim = 2;                              // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
      target_tensor_test_data.shape = target_shape_test_data;       // Set the shape of the input_tensor
      target_tensor_test_data.data = Anomaly_detection_test_data_label;// Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored


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

    // Input layer
    uint16_t input_layer_shape[] = {1, INPUTS};                     // Definition of the input layer shape (Must fit to the input tensor)
    ailayer_input_t input_layer;                                    // Creation of the AIfES input layer
      input_layer.input_dim = 2;                                    // Definition of the input dimension (Must fit to the input tensor)
      input_layer.input_shape = input_layer_shape;                  // Handover of the input layer shape

    // Dense layer (hidden layer)
    ailayer_dense_t dense_layer_1;                                  // Creation of the AIfES hidden dense layer
    dense_layer_1.neurons = NEURONS_1;                              // Number of neurons
    ailayer_sigmoid_t sigmoid_layer_1;                              // Sigmoid activation

    // Dense layer (hidden layer_2)
    ailayer_dense_t dense_layer_2;                                  // Creation of the AIfES hidden dense layer
    dense_layer_2.neurons = NEURONS_2;                              // Number of neurons
    ailayer_sigmoid_t sigmoid_layer_2;                              // Sigmoid activation function

    // Dense layer (hidden layer_3)
    ailayer_dense_t dense_layer_3;                                  // Creation of the AIfES hidden dense layer
    dense_layer_3.neurons = NEURONS_3;                              // Number of neurons
    ailayer_sigmoid_t sigmoid_layer_3;                              // Sigmoid activation function

    // Dense layer (hidden layer_4)
    ailayer_dense_t dense_layer_4;                                  // Creation of the AIfES hidden dense layer
    dense_layer_4.neurons = NEURONS_4;                              // Number of neurons
    ailayer_sigmoid_t sigmoid_layer_4;                              // Sigmoid activation function

    // Output dense layer
    ailayer_dense_t dense_layer_5;                                  // Creation of the AIfES ouput dense layer
    dense_layer_5.neurons = OUTPUTS;                                // Number of neurons
    ailayer_sigmoid_t sigmoid_layer_5;                              // Sigmoid activation function

    ailoss_crossentropy_t crossentropy_loss;                        // Loss: Crossentropy

    // --------------------------- Define the structure of the model ----------------------------

    aimodel_t model;                                                // AIfES model
    ailayer_t *x;                                                   // Layer object from AIfES, contains the layers

    // Passing the layers to the AIfES model
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    x = ailayer_dense_f32_default(&dense_layer_3, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_3, x);
    x = ailayer_dense_f32_default(&dense_layer_4, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_4, x);
    x = ailayer_dense_f32_default(&dense_layer_5, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_5, x);
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
    uint32_t batch_size = 200;
    float history[trials][EPOCHS+1][10];  //before training + epochs
    unsigned int seeds[trials];

    for (int t=0; t<trials; t++) {
        printf("\nStarting trial %i...\n",t+1);
        // ------------------------------- Initialize the parameters ------------------------------

        seeds[t] = time(NULL);
        //seeds[t] = 1634302329;
        srand(seeds[t]);

        aimath_f32_default_init_glorot_uniform(&dense_layer_1.weights);
        aimath_f32_default_init_zeros(&dense_layer_1.bias);
        dense_layer_1.maxError = 0;
        aimath_f32_default_init_glorot_uniform(&dense_layer_2.weights);
        aimath_f32_default_init_zeros(&dense_layer_2.bias);
        dense_layer_2.maxError = 0;
        aimath_f32_default_init_glorot_uniform(&dense_layer_3.weights);
        aimath_f32_default_init_zeros(&dense_layer_3.bias);
        dense_layer_3.maxError = 0;
        aimath_f32_default_init_glorot_uniform(&dense_layer_4.weights);
        aimath_f32_default_init_zeros(&dense_layer_4.bias);
        dense_layer_4.maxError = 0;
        aimath_f32_default_init_glorot_uniform(&dense_layer_5.weights);
        aimath_f32_default_init_zeros(&dense_layer_5.bias);
        dense_layer_5.maxError = 0;

        // ------------------------------------- Run the training ------------------------------------

        for(uint32_t i = 0; i < EPOCHS; i++)
        {
            float loss, acc;
            dense_layer_1.ratio = 0.0f;
            dense_layer_2.ratio = 0.0f;
            dense_layer_3.ratio = 0.0f;
            dense_layer_4.ratio = 0.0f;
            dense_layer_5.ratio = 0.0f;

            aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
            aialgo_inference_model(&model, &input_tensor, &output_tensor);
            predict_model_binary_acc(&acc, NUM_TRAIN_DATA, 0.5f, &Anomaly_detection_training_data_label, &output_data);
            if (i > 0) {
                float loss_test, test_acc;
                aialgo_inference_model(&model, &input_tensor_test_data, &output_tensor_test_data);
                predict_model_binary_acc(&test_acc, NUM_TEST_DATA, 0.5f, &Anomaly_detection_test_data_label, &output_test_data);
                aialgo_calc_loss_model_f32(&model, &input_tensor_test_data, &target_tensor_test_data, &loss_test);
                //reduction / normalization of the loss
                loss_test = loss_test / (OUTPUTS * NUM_TEST_DATA);
                history[t][i][8] = loss_test;
                history[t][i][9] = test_acc;
                printf("\rEpoch %d completed, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%", i, acc, test_acc);
            }
            else {
                printf("\rAccuracy before Training: %.2f%%", acc);

            }

            history[t][i][0] = loss/(OUTPUTS * NUM_TRAIN_DATA);
            history[t][i][1] = acc;

            // One epoch of training. Iterates through the whole data once
            #ifdef MONITOR_BP
            measure_backwards_pass_performance(&model, &input_tensor, &target_tensor, optimizer, batch_size);
            #else
			clock_t start = clock();
            aialgo_train_model_meProp(&model, &input_tensor, &target_tensor, optimizer, batch_size, MAX_BPR, MIN_BPR, DAMPING);
			history[t][i+1][2] = (float)(clock() - start)/CLOCKS_PER_SEC;
			history[t][i+1][3] = dense_layer_1.ratio/NUM_TRAIN_DATA;
			history[t][i+1][4] = dense_layer_2.ratio/NUM_TRAIN_DATA;
			history[t][i+1][5] = dense_layer_3.ratio/NUM_TRAIN_DATA;
			history[t][i+1][6] = dense_layer_4.ratio/NUM_TRAIN_DATA;
			history[t][i+1][7] = dense_layer_5.ratio/NUM_TRAIN_DATA;
			#endif

        }
        // training complete
        float loss, acc;
        aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
        aialgo_inference_model(&model, &input_tensor, &output_tensor);
        predict_model_binary_acc(&acc, NUM_TRAIN_DATA, 0.5f, &Anomaly_detection_training_data_label, &output_data);
        history[t][EPOCHS][0] = loss/(OUTPUTS * NUM_TRAIN_DATA);
        history[t][EPOCHS][1] = acc;

        float loss_test, test_acc;
        aialgo_inference_model(&model, &input_tensor_test_data, &output_tensor_test_data);
        predict_model_binary_acc(&test_acc, NUM_TEST_DATA, 0.5f, &Anomaly_detection_test_data_label, &output_test_data);
        aialgo_calc_loss_model_f32(&model, &input_tensor_test_data, &target_tensor_test_data, &loss_test);
        history[t][EPOCHS][8] = loss_test/(OUTPUTS * NUM_TEST_DATA);
        history[t][EPOCHS][9] = test_acc;

        printf("\rEpoch %d completed, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%\nFinished training\n\n", EPOCHS, acc, test_acc);
        printf("Trial %i completed\n",t+1);
    }


    create_history_data(str,history,seeds,trials);
    free(parameter_memory);
    free(memory_ptr);
	return 0;
}
