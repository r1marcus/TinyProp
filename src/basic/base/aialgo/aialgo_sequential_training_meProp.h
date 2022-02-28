/**
 * \file basic/base/aialgo/aialgo_sequential_training_meProp.h
 * \internal
 * \date 20.10.2020
 * \endinternal
 * \version 2.0alpha
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
 * \brief Functions required for the training of models
 * \details The functions target memory allocation/scheduling and the backpropagation for model training
 */

#ifndef AIALGO_SEQUENTIAL_TRAINING_MEPROP
#define AIALGO_SEQUENTIAL_TRAINING_MEPROP

#include "core/aifes_core.h"
#include "core/aifes_math.h"
#include "basic/base/aimath/aimath_basic.h"

/** @brief Perform the backward pass
 *
 * @param *model         The model
 * @param *target_data   The tensor containing the target data / labels
 * @param maxBpr         The maximal possible ratio (0 to 1) for backpropagation
 * @param minBpr         The minimal possible ratio (0 to 1) for backpropagation
 * @param damping        Rate at which the backpropagation gets smaller over layers
 */
void aialgo_backward_model_meProp(aimodel_t *model, aitensor_t *target_data, float maxBpr, float minBpr, float damping);

/** @brief Perform one training epoch on all data batches of the dataset using backpropagation
 *
 * Make shure to initialize the model (aialgo_compile_model()) and schedule the training memory
 * (for example with aialgo_schedule_training_memory()) and initialize the training memory
 * (aialgo_init_model_for_training()) before calling this function.
 *
 * Example: Training of an F32 model for multiple epochs
 * \code{.c}
 * int epochs = 100;
 * int batch_size = 4;
 * int print_interval = 10;
 *
 * float loss;
 * for(i = 0; i < epochs; i++)
 * {
 *     // One epoch of training. Iterates through the whole data once
 *     aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);

 *     // Calculate and print loss every print_interval epochs
 *     if(i % print_interval == 0)
 *     {
 *         aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
 *         printf("Epoch %5d: loss: %f\n", i, loss);
 *     }
 * }
 * \endcode
 *
 * @param *model            The model
 * @param *input_tensor     The tensor containing the input data
 * @param *target_tensor    The tensor containing the target data / labels
 * @param *optimizer        The optimizer that is used for training
 * @param batch_size        Size of a batch / Number of input vektors
 * @param maxBpr            The maximal possible ratio (0 to 1) for backpropagation
 * @param minBpr            The minimal possible ratio (0 to 1) for backpropagation
 * @param damping           Rate at which the backpropagation gets smaller over layers
 */
void aialgo_train_model_meProp(aimodel_t *model, aitensor_t *input_tensor, aitensor_t *target_tensor, aiopti_t *optimizer, uint32_t batch_size, float maxBpr, float minBpr, float damping);

#endif // AIALGO_SEQUENTIAL_TRAINING_MEPROP
