#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn

# CHANGE TINYPROP HYPERPARAMETERS HERE!
S_max = 1
S_min = 0
zeta = 1

class SparseLinear(torch.autograd.Function):  
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, info, bias=None):
        # Save inputs in context-object for later use in backwards
        ctx.save_for_backward(input, weight, bias) # these are differentiable
        ctx.info = info                          # non-differentiable argument
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        # input   [batchSize, in]
        # output  [batchSize, out]
        # weights [out, in]
        # bias    [out]
        grad_input = grad_weight = grad_bias = None
        
        # Calculate k (different across batches)
        Y = grad_output.abs().sum(1)       # Y[batchSize]
        if (torch.max(Y) > ctx.info.Y_max):     # Check if biggest Y of batch is bigger than recorded Y
            ctx.info.Y_max = torch.max(Y).item()
        bpr = (S_min + Y*(S_max-S_min)/(ctx.info.Y_max))*pow(zeta, ctx.info.layer_idx)
        #bpr = torch.ones((20)) 
        K = torch.round(grad_output.size(1)*bpr).detach().clone()  # K[batchSize]
        K.clamp(1, grad_output.size(1))
        # log in layer
        ctx.info.miniBatchBpr = torch.mean(bpr)
        ctx.info.miniBatchK = torch.mean(K)
        K = K.to(torch.int16)
        # create a sparse grad_output tensor. Since k is different across batches, the topK indices
        # must be assembled for each batch separately.
        col = []
        row = []
        val = []
        for batch,k in enumerate(K):
            _, indices = grad_output[batch].abs().topk(k)  # don't use return VALUES since they are abs()!
            col.append(indices)
            val.append(torch.index_select(grad_output[batch], -1, indices))
            row += indices.size(0) * [batch]
        col = torch.cat(col).detach()
        row = torch.Tensor(row)
        val = torch.cat(val)
        sparse = torch.sparse_coo_tensor(torch.vstack((row,col)), val, grad_output.size())
        
        # Do the usual bp stuff but use sparse matmul where needed
        if ctx.needs_input_grad[0]:
            grad_input = torch.sparse.mm(sparse, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.sparse.mm(sparse.t(), input)  # Gradients are zeroed each batch
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, None, grad_bias
    
class BackwardsInfo:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.Y_max = 0
        self.miniBatchBpr = 0
        self.miniBatchK = 0
        
class TinyPropLinear(nn.Module):
    def __init__(self, input_features, output_features, layer_idx, bias=True):
        super(TinyPropLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # Saving variables like this will pass it by REFERENCE, so changes 
        # made in backwards are reflected in layer
        self.info = BackwardsInfo(layer_idx) 
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()
            
    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.output_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # Here the custom linear function is applied
        return SparseLinear.apply(input, self.weight, self.info, self.bias)

