/**
 * \file basic/default/ailayer/ailayer_conv2d_default.h
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

#ifndef AILAYER_CONV2D_DEFAULT
#define AILAYER_CONV2D_DEFAULT

#include "basic/base/ailayer/ailayer_conv2d.h"
#include "basic/default/aimath/aimath_f32_default.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ailayer_conv2d ailayer_conv2d_f32_t;

ailayer_t *ailayer_conv2d_f32_default(ailayer_conv2d_f32_t *layer, ailayer_t *input_layer);

#ifdef __cplusplus
}
#endif

#endif // AILAYER_CONV2D_DEFAULT
