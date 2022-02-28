/**
 * \file basic/base/aialgo/aialgo_sequential_training_meProp.c
 * \version 2.0alpha
 * \date 20.10.2020
 * \copyright  Copyright (C) 2020-2021  Fraunhofer Institute for Microelectronic Circuits and Systems.
    All rights reserved.

    AIfES is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * \brief
 * \details
 */

#include "basic/base/aialgo/aialgo_sequential_training_meProp.h"
#include "basic/base/aialgo/aialgo_sequential_training.h"
#include "basic/base/aialgo/aialgo_sequential_inference.h"

// ToDo: Remove dependency
#include "basic/default/aimath/aimath_f32_default.h"

void aialgo_backward_model_meProp(aimodel_t *model, aitensor_t *target_data, float maxBpr, float minBpr, float damping)
{
	uint16_t i;
	ailayer_t *layer_ptr = model->output_layer;
	float bpr;

	model->loss->calc_delta(model->loss, target_data);
	int dense_counter = 0;
	for(i = 0; i < model->layer_count; i++)
	{
        #ifdef DEBUG_CHECKS
                if(layer_ptr->backward_meProp == 0){
                    printf("\nError: No backward function implementation in layer %d\n", i);
                    return;
                }
        #endif
		layer_ptr->backward_meProp(layer_ptr, minBpr, maxBpr, damping, dense_counter);
		if(layer_ptr->layer_type == ailayer_dense_type)
            dense_counter++;
		layer_ptr = layer_ptr->input_layer;
	}
	return;
}

void aialgo_train_model_meProp(aimodel_t *model, aitensor_t *input_tensor, aitensor_t *target_tensor, aiopti_t *optimizer, uint32_t batch_size, float maxBpr, float minBpr, float damping)
{
	uint32_t i;

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

	uint32_t batch_count = (uint32_t) (input_tensor->shape[0] / batch_size);
	uint32_t batch;
	for(batch = 0; batch < batch_count; batch++)
	{
		aialgo_zero_gradients_model(model, optimizer);
		for(i = 0; i < batch_size; i++)
		{
			input_batch.data = input_tensor->data + batch * input_multiplier * batch_size + i * input_multiplier;
			target_batch.data = target_tensor->data + batch * target_multiplier * batch_size + i * target_multiplier;


			//printf("Input batch [%d, %d, %d, %d]:\n", input_batch_shape[0], input_batch_shape[1], input_batch_shape[2], input_batch_shape[3]);
			//print_aitensor(&input_batch);

			aialgo_forward_model(model, &input_batch);
			aialgo_backward_model_meProp(model, &target_batch, minBpr, maxBpr, damping);
		}
		aialgo_update_params_model(model, optimizer);
	}
	return;
}
