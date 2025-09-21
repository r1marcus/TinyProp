import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.nn.common_types import _size_1_t, _size_2_t


# classes to hold TinyProp parameters on Net and Layer scope

class TinyPropParams:
    def __init__(self, S_min: float, S_max: float, zeta: float, number_of_layers: int):
        self.S_min = S_min
        self.S_max = S_max
        self.zeta = zeta
        self.number_of_layers = number_of_layers

class TinyPropLayer:
    def __init__(self, layerPosition: int):
        self.layerPosition = layerPosition
        self.Y_max = 0
        self.miniBatchBpr = 0
        self.miniBatchK = 0
        self.epochBpr = []
        self.epochK = []

    def BPR(self, params, Y):
        return (params.S_min + Y*(params.S_max-params.S_min)/(self.Y_max)) * (params.zeta**self.layerPosition)

    def selectGradients(self, grad_output, params):
        # assumes grad_output.shape = [batchSize, entries]

        # calculate bpr (different across batches)
        Y = grad_output.abs().sum(1)            # Y [batchSize]
        if (torch.max(Y) > self.Y_max):         # Check if biggest Y of batch is bigger than recorded Y
            self.Y_max = torch.max(Y).item()
        bpr = self.BPR(params, Y)               #bpr [batchSize]

        # calculate K [batchSize]
        K = torch.round(grad_output.size(1)*bpr)  # K [batchSize]
        K.clamp(1, grad_output.size(1))
        self.miniBatchBpr += torch.mean(bpr).item()
        self.miniBatchK += torch.mean(K).item()
        K = K.to(torch.int16)
        
        # create a sparse grad_output tensor. Since k is different across batches, the topK indices
        # must be assembled for each batch separately.
        idx = []    # indices of sparse entries [batch, element]
        val = []    # corresponding values, of size element
        for batch, k in enumerate(K):
            _, indices = grad_output[batch].abs().topk(k)  # don't use return VALUES since they are abs!
            t = torch.vstack((torch.zeros_like(indices) + batch, indices))
            idx.append(t)
            val.append(torch.index_select(grad_output[batch], -1, indices)) # select values from grad_output instead

        idx = torch.hstack(idx)
        val = torch.cat(val)
        return idx, val


#========== Helper functions ==========#

def _apply_tinyprop_mask(tp_info: "TinyPropLayer", grad_output: torch.Tensor, tp_params: TinyPropParams) -> torch.Tensor:
    """Apply the TinyProp gradient selection to the gradient tensor."""

    flattened = torch.flatten(grad_output, start_dim=1)
    indices, values = tp_info.selectGradients(flattened, tp_params)

    masked_flat = torch.zeros_like(flattened)
    if values.numel() > 0:
        masked_flat[indices[0], indices[1]] = values

    return masked_flat.view_as(grad_output)


#========== LINEAR ==========#

class SparseLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, tpParams: TinyPropParams, tpInfo: TinyPropLayer, bias=None):      # bias is an optional argument
        # Save inputs in context-object for later use in backwards. Alternatively, this part could be done in a setup_context() method
        ctx.save_for_backward(input, weight, bias) # these are differentiable
        # non-differentiable arguments, directly stored on ctx
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo

        # Do the mathematical operations associated with forwards
        return F.linear(input, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Unpack saved tensors. NEVER modify these in the backwards function!
        input, weight, bias = ctx.saved_tensors
        # input   [batchSize, in]
        # output  [batchSize, out]
        # weights [out, in]
        # bias    [out]

        # Initialize all gradients w.r.t. inputs to None
        grad_input = grad_weight = grad_bias = None

        # This is the TinyProp part:
        indices, values = ctx.tpInfo.selectGradients(grad_output, ctx.tpParams)
        sparse = torch.sparse_coo_tensor(indices, values, grad_output.size())
        
        # Do the usual bp stuff but use sparse matmul on grad_input and grad_weight
        if ctx.needs_input_grad[0]:
            grad_input = torch.sparse.mm(sparse, weight)      #[batchSize, in]
        if ctx.needs_input_grad[1]:
            grad_weight = torch.sparse.mm(sparse.t(), input)  # Gradients are zeroed each batch, batch dimension is reduced in operation -> [out, in]
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, None, None, grad_bias
    

# Create TinyProp verion of Linear by extending it. This way it integrates seemlessly into existing code
class TinyPropLinear(TinyPropLayer, nn.Linear):
    def __init__(self, in_features: int, out_features: int, tinyPropParams: TinyPropParams, layer_number: int, bias: bool=True, device=None, dtype=None):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        
        # Saving variables like this will pass it by REFERENCE, so changes 
        # made in backwards are reflected in layer
        self.tpParams = tinyPropParams

    def forward(self, input):
        # Here the custom linear function is applied
        return SparseLinear.apply(input, self.weight, self.tpParams, self, self.bias)
    

#========== CONVOLUTION 1D ==========#

class SparseConv1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, padding_mode,
                _reversed_padding_repeated_twice, tpParams: TinyPropParams, tpInfo: TinyPropLayer):
        ctx.save_for_backward(input, weight, bias)

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo

        if padding_mode != 'zeros':
            return F.conv1d(F.pad(input, _reversed_padding_repeated_twice, mode=padding_mode),
                            weight, bias, stride, 0, dilation, groups)
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        masked_grad = _apply_tinyprop_mask(ctx.tpInfo, grad_output, ctx.tpParams)

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv1d_input(input.shape, weight, masked_grad, ctx.stride,
                                                    ctx.padding, ctx.dilation, ctx.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv1d_weight(input, weight.shape, masked_grad, ctx.stride,
                                                      ctx.padding, ctx.dilation, ctx.groups)
        if bias is not None and ctx.needs_input_grad[2]:
            sum_dims = (0,) + tuple(range(2, masked_grad.dim()))
            grad_bias = masked_grad.sum(dim=sum_dims)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class TinyPropConv1d(TinyPropLayer, nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 tinyPropParams: TinyPropParams,
                 layer_number: int,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)

        self.tpParams = tinyPropParams

    def forward(self, input):
        return SparseConv1d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups, self.padding_mode, self._reversed_padding_repeated_twice,
                                  self.tpParams, self)


#========== CONVOLUTION 2D ==========#

class SparseConv2d(torch.autograd.Function):
    # keep in mind that convolution operations DO NOT reduce the batchSize (in contrast to matmul)!

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, padding_mode, _reversed_padding_repeated_twice, tpParams: TinyPropParams, tpInfo: TinyPropLayer):
        # Save inputs in context-object for later use in backwards. Alternatively, this part could be done in a setup_context() method
        ctx.save_for_backward(input, weight, bias) # these are differentiable
        # non-differentiable arguments, directly stored on ctx
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo

        # Do the 2d convolution exactly as normal Conv2d layer
        # conv2d does not reduce the batch-dimension -> [batchSize, out, width, height]
        if padding_mode != 'zeros':
            return F.conv2d(F.pad(input, _reversed_padding_repeated_twice, mode=padding_mode),
                               weight, bias, stride, nn.modules.utils._pair(0), dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    
    @staticmethod
    def backward(ctx, grad_output):
        # Unpack saved tensors. NEVER modify these in the backwards function!
        input, weight, bias = ctx.saved_tensors
        # input   [batchSize, in, width, height]
        # output  [batchSize, out, width, height]
        # weights [batchSize, out, in/groups, width, height]
        # bias    [batchSize, out]

        # Initialize all gradients w.r.t. inputs to None
        grad_input = grad_weight = grad_bias = None

        masked_grad = _apply_tinyprop_mask(ctx.tpInfo, grad_output, ctx.tpParams)

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, masked_grad, ctx.stride,
                                                    ctx.padding, ctx.dilation, ctx.groups)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, masked_grad, ctx.stride,
                                                      ctx.padding, ctx.dilation, ctx.groups)

        if bias is not None and ctx.needs_input_grad[2]:
            sum_dims = (0,) + tuple(range(2, masked_grad.dim()))
            grad_bias = masked_grad.sum(dim=sum_dims)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None
    
    
# Create TinyProp verion of Conv2d by extending it. This way it integrates seemlessly into existing code
class TinyPropConv2d(TinyPropLayer, nn.Conv2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        tinyPropParams: TinyPropParams,
        layer_number: int,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)
        
        # Saving variables like this will pass it by REFERENCE, so changes 
        # made in backwards are reflected in layer
        self.tpParams = tinyPropParams

    def forward(self, input):
        # Here the custom conv2d function is applied
        return SparseConv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                                  self.padding_mode, self._reversed_padding_repeated_twice, self.tpParams, self)
    

#========== TRAINING ==========#

def trainOneEpoch(device, model, optimizer, loss_function, train_loader, epoch, print_interval=10):
    model.train()   # set the model to train mode
    batch_idx = 0
    running_loss = 0
    running_accuracy = 0

    # loop over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        idx_predicted = torch.max(output.data, 1)[1]    # get index of predicted class (max value)
        running_accuracy += (idx_predicted == target).sum().item()

        if batch_idx % print_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()), end ='\r')
    print('Train Epoch: {} completed            '.format(epoch))

    # log mean epoch bpr and k for each TinyPropLayer automatically
    for layer in model.children():
        if isinstance(layer, TinyPropLayer):
            layer.epochBpr.append(layer.miniBatchBpr/batch_idx)
            layer.miniBatchBpr = 0
            layer.epochK.append(layer.miniBatchK/batch_idx)
            layer.miniBatchK = 0

    return running_loss/batch_idx, 100*running_accuracy/len(train_loader.dataset)


def evaluate(device, model, loss_function, test_loader):
    model.eval()  # set the model to evaluation mode
    batch_idx = 0
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):  #loop through batches
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            running_loss += loss_function(output, target).item()
            idx_predicted = torch.max(output.data, 1)[1]    # get index of predicted class (max value)
            running_acc += (idx_predicted == target).sum().item()

    test_loss = running_loss / batch_idx
    test_accuracy = 100. * running_acc / len(test_loader.dataset)
    print('Test  Eval : Avg.loss: {:.4f},  Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, running_acc, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy
