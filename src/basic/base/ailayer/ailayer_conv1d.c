/**
 * \file basic/base/ailayer/ailayer_conv1d.c
 * \version 2.0alpha
 * \date 27.05.2024
 * \copyright  Copyright (C) 2020-2024  Fraunhofer Institute for Microelectronic Circuits and Systems.
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
 */

#include "basic/base/ailayer/ailayer_conv1d.h"
#include "basic/base/aimath/aimath_basic.h"

const aicore_layertype_t ailayer_conv1d_type_s = {
#ifdef AIDEBUG_PRINT_MODULE_SPECS
    .name = "Conv1D",
    .print_specs = ailayer_conv1d_print_specs
#else
    .name = 0,
    .print_specs = 0
#endif
};
const aicore_layertype_t *ailayer_conv1d_type = &ailayer_conv1d_type_s;

ailayer_t *ailayer_conv1d(ailayer_conv1d_t *layer, ailayer_t *input_layer)
{
    layer->requires_grad = 0x03; /* weights and bias enabled by default */
    layer->base.layer_type = ailayer_conv1d_type;

    layer->base.input_layer = input_layer;
    input_layer->output_layer = &(layer->base);

    layer->base.layer_configuration = layer;
    layer->base.result.dtype = layer->result_dtype;
    layer->base.result.dim = 3;
    layer->base.result.shape = layer->result_shape;

    layer->base.deltas.dtype = layer->result_dtype;
    layer->base.deltas.dim = input_layer->result.dim;
#ifdef DEBUG_CHECKS
    if(input_layer->result.dim != 3)
    {
        LOG_E("Conv1D: input tensor must have 3 dimensions.\n");
        return 0;
    }
#endif
    layer->base.deltas.shape = layer->deltas_shape;

    uint8_t i;
    for(i = 0; i < input_layer->result.dim && i < 3; i++){
        layer->deltas_shape[i] = input_layer->result.shape[i];
    }

    layer->weights.dim = 3;
    layer->weights.dtype = layer->weights_dtype;
    layer->weights.shape = layer->weights_shape;
    layer->weights.shape[0] = layer->out_channels;
#ifdef DEBUG_CHECKS
    if(layer->groups == 0){
        LOG_E("Conv1D: groups must be greater than zero.\n");
        return 0;
    }
    if(input_layer->result.shape[1] % layer->groups != 0){
        LOG_E("Conv1D: input channels not divisible by groups.\n");
        return 0;
    }
    if(layer->out_channels % layer->groups != 0){
        LOG_E("Conv1D: output channels not divisible by groups.\n");
        return 0;
    }
#endif
    layer->weights.shape[1] = input_layer->result.shape[1] / layer->groups;
    layer->weights.shape[2] = layer->kernel_size;

    layer->bias.dim = 1;
    layer->bias.dtype = layer->bias_dtype;
    layer->bias.shape = layer->bias_shape;
    layer->bias.shape[0] = layer->out_channels;

    layer->base.forward = ailayer_conv1d_forward;
    layer->base.backward = ailayer_conv1d_backward;
    layer->base.backward_meProp = ailayer_conv1d_backward_meProp;

    layer->base.calc_result_shape = ailayer_conv1d_calc_result_shape;
    layer->base.sizeof_paramem = ailayer_conv1d_sizeof_paramem;
    layer->base.set_paramem = ailayer_conv1d_set_paramem;
    layer->base.sizeof_trainmem = ailayer_conv1d_sizeof_trainmem;
    layer->base.set_trainmem = ailayer_conv1d_set_trainmem;

    layer->base.get_result_bound = 0;

    layer->base.trainable_params_count = 2;
    layer->base.trainable_params = layer->trainable_params;
    layer->base.gradients = layer->gradients;
    layer->base.optimem = layer->optimem;

    layer->trainable_params[0] = &(layer->weights);
    layer->trainable_params[1] = &(layer->bias);

    return &(layer->base);
}

void ailayer_conv1d_forward(ailayer_t *self)
{
    ailayer_conv1d_t *layer = (ailayer_conv1d_t *)(self->layer_configuration);
    aitensor_t *input_tensor = &(self->input_layer->result);
    aitensor_t *result_tensor = &(self->result);

    layer->conv(input_tensor, &(layer->weights), &(layer->bias),
                layer->stride, layer->padding, layer->dilation, layer->groups,
                result_tensor);
}

void ailayer_conv1d_backward(ailayer_t *self)
{
    ailayer_conv1d_t *layer = (ailayer_conv1d_t *)(self->layer_configuration);
    aitensor_t *delta_in = &(self->deltas);
    aitensor_t *delta_out = &(self->output_layer->deltas);
    aitensor_t *x_in = &(self->input_layer->result);

    if(layer->requires_grad & 0x01){
        layer->conv_weight_grad(delta_out, x_in,
                                layer->stride, layer->padding, layer->dilation, layer->groups,
                                layer->gradients[0]);
    }
    if(layer->requires_grad & 0x02){
        layer->conv_bias_grad(delta_out, layer->gradients[1]);
    }

    layer->conv_input_grad(delta_out, &(layer->weights),
                           layer->stride, layer->padding, layer->dilation, layer->groups,
                           delta_in);
}

void ailayer_conv1d_backward_meProp(ailayer_t *self, float maxBpr, float minBpr, float damping, int dense_counter)
{
    (void)maxBpr;
    (void)minBpr;
    (void)damping;
    (void)dense_counter;
    ailayer_conv1d_backward(self);
}

void ailayer_conv1d_calc_result_shape(ailayer_t *self)
{
    ailayer_conv1d_t *layer = (ailayer_conv1d_t *)(self->layer_configuration);
    aitensor_t *x_in = &(self->input_layer->result);

    uint16_t batch = x_in->shape[0];
    uint16_t in_length = x_in->shape[2];

    int32_t numerator = (int32_t)in_length + 2 * (int32_t)layer->padding -
                        (int32_t)layer->dilation * ((int32_t)layer->kernel_size - 1) - 1;
    uint16_t out_length = (uint16_t)(numerator / layer->stride + 1);

    self->result.shape[0] = batch;
    self->result.shape[1] = layer->out_channels;
    self->result.shape[2] = out_length;

    layer->deltas_shape[0] = batch;
    layer->deltas_shape[1] = x_in->shape[1];
    layer->deltas_shape[2] = in_length;
}

uint32_t ailayer_conv1d_sizeof_paramem(const ailayer_t *self)
{
    const ailayer_conv1d_t *layer = (const ailayer_conv1d_t *)(self->layer_configuration);
    uint32_t memory = 0;

    memory += layer->weights_dtype->tensor_params_size;
    memory += layer->out_channels * layer->weights.shape[1] * layer->kernel_size *
              aimath_sizeof_dtype(layer->weights_dtype);

    memory += layer->bias_dtype->tensor_params_size;
    memory += layer->out_channels * aimath_sizeof_dtype(layer->bias_dtype);

    return memory;
}

void ailayer_conv1d_set_paramem(ailayer_t *self, void *memory_ptr)
{
    uint32_t address_counter = 0;
    ailayer_conv1d_t *layer = (ailayer_conv1d_t *)(self->layer_configuration);

    layer->weights.tensor_params = memory_ptr + address_counter;
    address_counter += layer->weights_dtype->tensor_params_size;
    layer->weights.dim = 3;
    layer->weights.dtype = layer->weights_dtype;
    layer->weights.shape = layer->weights_shape;
    layer->weights.shape[0] = layer->out_channels;
    layer->weights.shape[1] = self->input_layer->result.shape[1] / layer->groups;
    layer->weights.shape[2] = layer->kernel_size;
    layer->weights.data = memory_ptr + address_counter;
    address_counter += aimath_sizeof_tensor_data(&(layer->weights));

    layer->bias.tensor_params = memory_ptr + address_counter;
    address_counter += layer->bias_dtype->tensor_params_size;
    layer->bias.dim = 1;
    layer->bias.dtype = layer->bias_dtype;
    layer->bias.shape = layer->bias_shape;
    layer->bias.shape[0] = layer->out_channels;
    layer->bias.data = memory_ptr + address_counter;

    layer->trainable_params[0] = &(layer->weights);
    layer->trainable_params[1] = &(layer->bias);
}

uint32_t ailayer_conv1d_sizeof_trainmem(const ailayer_t *self)
{
    const ailayer_conv1d_t *layer = (const ailayer_conv1d_t *)(self->layer_configuration);
    uint32_t memory = 0;

    memory += aimath_sizeof_tensor(&(layer->weights));
    memory += aimath_sizeof_tensor(&(layer->bias));

    return memory;
}

void ailayer_conv1d_set_trainmem(ailayer_t *self, void *memory_ptr)
{
    uint32_t address_counter = 0;
    ailayer_conv1d_t *layer = (ailayer_conv1d_t *)(self->layer_configuration);

    self->gradients[0] = memory_ptr + address_counter;
    address_counter += sizeof(aitensor_t);
    self->gradients[0]->dim = layer->weights.dim;
    self->gradients[0]->dtype = layer->weights.dtype;
    self->gradients[0]->shape = layer->weights.shape;
    self->gradients[0]->data = memory_ptr + address_counter;
    address_counter += aimath_sizeof_tensor_data(self->gradients[0]);
    self->gradients[0]->tensor_params = memory_ptr + address_counter;
    address_counter += aimath_sizeof_tensor_params(self->gradients[0]);

    self->gradients[1] = memory_ptr + address_counter;
    address_counter += sizeof(aitensor_t);
    self->gradients[1]->dim = layer->bias.dim;
    self->gradients[1]->dtype = layer->bias.dtype;
    self->gradients[1]->shape = layer->bias.shape;
    self->gradients[1]->data = memory_ptr + address_counter;
    address_counter += aimath_sizeof_tensor_data(self->gradients[1]);
    self->gradients[1]->tensor_params = memory_ptr + address_counter;
    address_counter += aimath_sizeof_tensor_params(self->gradients[1]);
}

#ifdef AIDEBUG_PRINT_MODULE_SPECS
void ailayer_conv1d_print_specs(const ailayer_t *self, int (*print)(const char *format, ...))
{
    const ailayer_conv1d_t *layer = (const ailayer_conv1d_t *)(self->layer_configuration);
    print("out_channels: %lu, kernel_size: %lu, stride: %lu", (unsigned long)layer->out_channels,
          (unsigned long)layer->kernel_size, (unsigned long)layer->stride);
}
#endif
