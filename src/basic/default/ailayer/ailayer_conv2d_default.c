/**
 * \file basic/default/ailayer/ailayer_conv2d_default.c
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

#include "basic/default/ailayer/ailayer_conv2d_default.h"

ailayer_t *ailayer_conv2d_f32_default(ailayer_conv2d_f32_t *layer, ailayer_t *input_layer)
{
    layer->result_dtype = aif32;
    layer->weights_dtype = aif32;
    layer->bias_dtype = aif32;

    layer->conv = aimath_f32_default_conv2d_forward;
    layer->conv_input_grad = aimath_f32_default_conv2d_input_grad;
    layer->conv_weight_grad = aimath_f32_default_conv2d_weight_grad;
    layer->conv_bias_grad = aimath_f32_default_conv_bias_grad;

    return ailayer_conv2d(layer, input_layer);
}
