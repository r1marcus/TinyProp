/**
 * \file basic/base/ailayer/ailayer_conv2d.h
 * \internal
 * \date 27.05.2024
 * \endinternal
 * \version 2.0alpha
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

#ifndef AILAYER_CONV2D
#define AILAYER_CONV2D

#include "core/aifes_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Forward declaration of the Conv2D layer structure. */
typedef struct ailayer_conv2d ailayer_conv2d_t;

/**
 * @brief General Conv2D layer structure.
 */
struct ailayer_conv2d {
    ailayer_t base; /**< Inherited field members from general ailayer struct. */
    const aimath_dtype_t *result_dtype; /**< Data type of the inference result values. */
    const aimath_dtype_t *weights_dtype; /**< Data type of the weights. */
    const aimath_dtype_t *bias_dtype; /**< Data type of the bias. */

    /** @name Layer configuration */
    ///@{
    uint16_t out_channels; /**< Number of output feature maps. */
    uint16_t kernel_height; /**< Kernel height. */
    uint16_t kernel_width; /**< Kernel width. */
    uint16_t stride_height; /**< Stride along the height dimension. */
    uint16_t stride_width; /**< Stride along the width dimension. */
    uint16_t padding_height; /**< Padding applied along the height dimension. */
    uint16_t padding_width; /**< Padding applied along the width dimension. */
    uint16_t dilation_height; /**< Dilation along the height dimension. */
    uint16_t dilation_width; /**< Dilation along the width dimension. */
    uint16_t groups; /**< Number of blocked connections. */
    ///@}

    /** @name Trainable parameters */
    ///@{
    aitensor_t weights; /**< Tensor containing the convolution kernels. */
    aitensor_t bias; /**< Tensor containing the bias. */

    uint16_t result_shape[4]; /**< Result tensor shape (batch, channels, height, width). */
    uint16_t deltas_shape[4]; /**< Delta tensor shape. */
    uint16_t weights_shape[4]; /**< Weight tensor shape (out_channels, in_channels/groups, kernel_h, kernel_w). */
    uint16_t bias_shape[1]; /**< Bias tensor shape (out_channels). */

    uint8_t requires_grad; /**< Bit mask to control gradient calculation (bit0: weights, bit1: bias). */

    aitensor_t *trainable_params[2]; /**< Pointers to trainable parameter tensors. */
    aitensor_t *gradients[2]; /**< Gradient tensors (same ordering as trainable_params). */
    void *optimem[2]; /**< Memory used by the training optimizer. */
    ///@}

    /** @name Math functions */
    ///@{
    void (*conv)(const aitensor_t *input, const aitensor_t *weights, const aitensor_t *bias,
                 uint16_t stride_height, uint16_t stride_width,
                 uint16_t padding_height, uint16_t padding_width,
                 uint16_t dilation_height, uint16_t dilation_width,
                 uint16_t groups, aitensor_t *output);
    void (*conv_input_grad)(const aitensor_t *delta_out, const aitensor_t *weights,
                            uint16_t stride_height, uint16_t stride_width,
                            uint16_t padding_height, uint16_t padding_width,
                            uint16_t dilation_height, uint16_t dilation_width,
                            uint16_t groups, aitensor_t *delta_in);
    void (*conv_weight_grad)(const aitensor_t *delta_out, const aitensor_t *input,
                             uint16_t stride_height, uint16_t stride_width,
                             uint16_t padding_height, uint16_t padding_width,
                             uint16_t dilation_height, uint16_t dilation_width,
                             uint16_t groups, aitensor_t *d_weights);
    void (*conv_bias_grad)(const aitensor_t *delta_out, aitensor_t *d_bias);
    ///@}
};

/** @brief Conv2D layer type indicator. */
extern const aicore_layertype_t *ailayer_conv2d_type;

ailayer_t *ailayer_conv2d(ailayer_conv2d_t *layer, ailayer_t *input_layer);
void ailayer_conv2d_forward(ailayer_t *self);
void ailayer_conv2d_backward(ailayer_t *self);
void ailayer_conv2d_backward_meProp(ailayer_t *self, float maxBpr, float minBpr, float damping, int dense_counter);
void ailayer_conv2d_calc_result_shape(ailayer_t *self);
uint32_t ailayer_conv2d_sizeof_paramem(const ailayer_t *self);
void ailayer_conv2d_set_paramem(ailayer_t *self, void *memory_ptr);
uint32_t ailayer_conv2d_sizeof_trainmem(const ailayer_t *self);
void ailayer_conv2d_set_trainmem(ailayer_t *self, void *memory_ptr);

#ifdef AIDEBUG_PRINT_MODULE_SPECS
void ailayer_conv2d_print_specs(const ailayer_t *self, int (*print)(const char *format, ...));
#endif

#ifdef __cplusplus
}
#endif

#endif // AILAYER_CONV2D
