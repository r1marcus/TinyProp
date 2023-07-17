// Used to monitor the time used for a backwards pass ONLY by doing it multiple times and estimating the average time
void measure_backwards_pass_performance(aimodel_t *model, aitensor_t *input_tensor, aitensor_t *target_tensor, aiopti_t *optimizer, uint32_t batch_size, float maxBpr, float minBpr, float damping, uint32_t numBackwardsPasses)
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
			for (int i=0; i<numBackwardsPasses; i++) {
                aialgo_backward_model_meProp(model, &target_batch, maxBpr, minBpr, damping);
			}
			elapsedTime += (float)(clock() - start)/CLOCKS_PER_SEC;
		}
		aialgo_update_params_model(model, optimizer);
	}
	printf("backwards pass done %i times\n",numBackwardsPasses*count);
	printf("elapsed time: %f\n",elapsedTime);
	printf("resulting in %f us/bp\n\n",elapsedTime*1000000/numBackwardsPasses/count);
	return;
}
