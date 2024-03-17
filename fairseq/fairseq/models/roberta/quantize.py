from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_MULTIHEAD = (1, -1) # used in quantize_wrap , in model.py

USE_HOOKS=False

# TODO: multihead input is (time, batch, channel); alternative way to collapse dimension for range/zero calculation?
#       for now, following: 
#           flatten_dims=(1, -1) => (time, batch*channel) => min(axis=1) => (time, 1) => min again => single value
#            i.e., for each timestep, there is a min; then get min among those

# @ MEMO: MultiHeadAttention module from fairseq:
    # in forward method, all linear layers use (q,k,v) as input;
    # but uses F.multi_head_attention_forward with only the weights/biases q_proj, k_proj, v_proj 
    # so even if quantize output of linear layers, there are instances where the output is not quantized,
    # because the forward method of q_proj, k_proj, v_proj is not called 

# TODO: change to register hook implementation


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        # print(x_flat.shape)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)

def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False):
    if qparams:
        if qparams.num_bits:
            return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic, inplace)
    elif num_bits:
        return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace)

    return x



class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2. ** (num_bits - 1)) if signed else 0.
        qmax = qmin + 2. ** num_bits - 1.
        scale = qparams.range / (qmax - qmin)

        min_scale = torch.tensor(1e-8).expand_as(scale).cuda()
        scale = torch.max(scale, min_scale)

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None

class QuantMeasure(nn.Module):

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN, inplace=False, 
                 dequantize=True, stochastic=False, momentum=0.9, measure=False,
                 reduce_dim=0, reduce_type='extreme', dynamic_qparams = True):

        super(QuantMeasure, self).__init__()        
        # keep track of zero point and range for quantization
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))

        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.reduce_dim = reduce_dim
        self.reduce_type = reduce_type
        self.dynamic_qparams = dynamic_qparams
        self.evaluation = False

    # TODO: simplify logic for measure/evaluation
    # post-quantization, collect stats during TRAINING => `measure` = TRUE (`self.evaluation` updates dynamically)
    # aware quantization => `measure` = FALSE

    # evaluate with quantization => model.eval(), measure = FALSE
    # evaluate without quantization => model.eval(), measure = TRUE
    
    # post-quantization, collect stats with EVAL => measure = TRUE, self.evaluation = FALSE (set it outside of the class)
    def forward(self, input, num_bits, qparams=None):
        #=======================================================================
        # calculte qparams
        #=======================================================================
        # if training or measuring, calculate qparams and update avg range and zero point
        if (self.training or self.measure):
            if not self.evaluation: # dont update for test passes during training           
                # calculate range and zero point
                if qparams is None:
                    qparams = calculate_qparams(input, num_bits=num_bits, flatten_dims=self.flatten_dims, 
                                                reduce_dim=self.reduce_dim, reduce_type=self.reduce_type)
                    # print(qparams.range.shape)
                # update running zero point and range
                with torch.no_grad():
                    if self.measure:
                        # adjust momentum if measure mode
                        momentum = self.num_measured / (self.num_measured + 1)
                        self.num_measured += 1
                    else:
                        # otherwise, use fixed momentum for updates
                        momentum = self.momentum

                    # Update the zero point with a weighted average of the new and existing values
                    self.running_zero_point.mul_(momentum).add_(qparams.zero_point * (1 - momentum))

                    # Update the range with a weighted average of the new and existing values
                    self.running_range.mul_(momentum).add_(qparams.range * (1 - momentum))

        # If evaluation mode (not measuring), compute qparams with avg range and zero point
        else:
            if self.dynamic_qparams:
                qparams = QParams(range=self.running_range, zero_point=self.running_zero_point, num_bits=num_bits)
            else:
                qparams = calculate_qparams(input, num_bits=num_bits, flatten_dims=self.flatten_dims, 
                                            reduce_dim=self.reduce_dim, reduce_type=self.reduce_type)

        #=======================================================================
        # quantize
        #=======================================================================
        # If in measurement mode, return the input directly without quantizing 
           # obs: during training, with measure=True, does NOT quantize => post quantization
        if self.measure:
            return input

        # Otherwise, quantize # obs: self.training=True: (1) update avg range,zero (above) and (2) quantize with current range,zero
        else:
            # print("\n\n\nQuantized during training!\n\n\n")
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                                stochastic=self.stochastic, inplace=self.inplace)
            return q_input

    def eval(self):
        super(QuantMeasure, self).eval()
        self.evaluation = True

    def train(self, mode=True):
        super(QuantMeasure, self).train(mode)
        if mode:
            # Training mode
            self.evaluation = False
        else:
            # Evaluation mode
            self.evaluation = True
        


class QuantizedLayer(nn.Module):
    def __init__(self, layer, n_bit_weight, n_bit_input, measure_w, measure_in,
                 input_idxs=None, input_keys=None, sublayers = False, momentum=0.9, layers_quantize=(nn.Linear), 
                 dynamic_qinput=True, dynamic_qweight=False, quantize_weights=True, quantize_inputs=True,
                 shape_measure_in=(1,1,1), shape_measure_w=(1,1), shape_measure_out=(1,1,1), 
                 flatten_dims_input=_DEFAULT_FLATTEN, flatten_dims_weight=_DEFAULT_FLATTEN,
                 quant_out=False):

        super(QuantizedLayer, self).__init__()
        self.layer = layer
        self.n_bit_weight, self.n_bit_input = n_bit_weight, n_bit_input
        self.measure_w, self.measure_in = measure_w, measure_in
        self.layers_quantize = layers_quantize
        self.momentum = momentum
        self.dynamic_qweight = dynamic_qweight
        self.dynamic_qinput = dynamic_qinput
        self.quantize_weights = quantize_weights
        self.quantize_inputs = quantize_inputs
        self.quantize_output = quant_out

        self.input_args_kwargs = [item for sublist in [input_idxs, input_keys] if sublist is not None for item in sublist]
        # define layers to quantize
        if sublayers:
            self.mods_quantize = [mod for mod in self.layer.children() if isinstance(mod, self.layers_quantize)]
        else:
            self.mods_quantize = [self.layer]

        # list for easy access to quantizers
        self.weight_quantizers = []
        self.input_quantizers = []
        self.quant_outs = []

        #FIXME - workaround after discovered quantization had to be different
        # this allows for fairseq's MultiHeadAttention to access biases and weights of its children
        if len(self.mods_quantize) == 1:
            self.bias = layer.bias
            self.weight = layer.weight

        self.set_quantizers(shape_measure_in, shape_measure_w, flatten_dims_input, flatten_dims_weight, shape_measure_out)

        if USE_HOOKS and self.quantize_weights:
            self.register_hooks()
  
    def set_quantizers(self, shape_measure_in, shape_measure_w, flatten_dims_input, flatten_dims_weight, shape_measure_out=(1,1,1)):
        # Set quantizer for inputs
        # @memo: allowed possibility for multiple inputs because MultiHeadAttention pass (q,k,v) directly to pytorch F.multi_head_attention
        # so we have to quantize it and not the inputs to the v_proj, k_proj, q_proj
        if self.quantize_inputs:
            for i, _ in enumerate(self.input_args_kwargs):
                quantizer_name = f"quant_input_{i}"
                quantizer = QuantMeasure(shape_measure=shape_measure_in, flatten_dims=flatten_dims_input,
                                            measure=self.measure_in, momentum=self.momentum, inplace=False, 
                                            dynamic_qparams=self.dynamic_qinput)
                setattr(self, quantizer_name, quantizer)
                self.input_quantizers.append(quantizer)

        if self.quantize_weights:
            for i, mod in enumerate(self.mods_quantize):
                if mod.weight is not None:
                    quantizer_name = f"quant_weight_{i}"
                    shape_measure_w = (mod.weight.shape[0],1)
                    quantizer = QuantMeasure(shape_measure=shape_measure_w, flatten_dims=flatten_dims_weight,
                                            measure=self.measure_w, momentum=self.momentum, inplace=False,
                                            reduce_dim=None, reduce_type='mean', dynamic_qparams=self.dynamic_qweight)
                    setattr(self, quantizer_name, quantizer)
                    self.weight_quantizers.append(quantizer)

        if self.quantize_output:
            # TODO: add args for output quantizers (e.g., flatten_dims, reduce_dim, reduce_type)
            self.quant_out =  QuantMeasure(shape_measure=shape_measure_in, flatten_dims=flatten_dims_input,
                                        measure=shape_measure_out, momentum=self.momentum, inplace=False, 
                                        dynamic_qparams=self.dynamic_qinput)

    def forward(self, *args, **kwargs):
        # print(f"\n\n\nForward of {self.layer.__class__.__name__}")
        # print("Parent layer", self.parent)
        
        #======================================================================
        # quantize inputs
        #======================================================================
        new_args = [None]*len(args); new_kwargs = {}
        i = 0
        if self.quantize_inputs:
            for arg_idx, arg in enumerate(args):
                if arg_idx in self.input_args_kwargs:
                    # print(f"Quantizing arg {arg_idx} with {self.n_bit_input} bits")
                    new_args[arg_idx] = self.input_quantizers[i](arg, self.n_bit_input)
                    i+=1
                else:
                    new_args[arg_idx] = arg

            for k, v in kwargs.items():
                if k in self.input_args_kwargs:
                    # print(f"Quantizing kwarg {k} with {self.n_bit_input} bits")
                    new_kwargs[k] = self.input_quantizers[i](v, self.n_bit_input)
                    i+=1
                else:
                    new_kwargs[k] = v
        #======================================================================
        # quantize weights
        #======================================================================
        if not USE_HOOKS and self.quantize_weights:
            orig_weights, orig_biases = [], []
            for i, mod in enumerate(self.mods_quantize):
                # print(f"Quantizing weights of {self.layer.__class__.__name__}")
                if hasattr(mod, 'weight') and mod.weight is not None:
                    orig_weights.append(mod.weight.data.clone())
                    mod.weight.data = self.weight_quantizers[i](mod.weight, self.n_bit_weight)

                if hasattr(mod, 'bias') and mod.bias is not None:
                    if not self.weight_quantizers[i].measure:                    
                        orig_biases.append(mod.bias.data.clone())
                        mod.bias.data = quantize(mod.bias, num_bits=self.n_bit_weight, flatten_dims=(0, -1))

        # FIXME check if internal bias reference is being update too
        # assert torch.allclose(self.bias.data, self.mods_quantize[0].bias.data)

        # forward pass
        outputs = self.layer(*new_args, **new_kwargs)
        # print(self.quantize_output)
        if self.quantize_output:
            # print(f"Quantizing output of {self.layer.__class__.__name__})")
            outputs = self.quant_out(outputs, self.n_bit_input)

        # restore weights and biases
        if not USE_HOOKS and self.quantize_weights:
            for i, mod in enumerate(self.mods_quantize):
                if hasattr(mod, 'weight') and mod.weight is not None:
                    mod.weight.data= orig_weights[i]
                if hasattr(mod, 'bias') and mod.bias is not None:
                    if not self.weight_quantizers[i].measure:                    
                        mod.bias.data = orig_biases[i]
        return outputs
      
    def register_hooks(self):
        self.hooks = []
        for i, mod in enumerate(self.mods_quantize):
            # qhook = QuantizationHooks(getattr(self, f"quant_weight_{i}"), self.n_bit_weight)
            qhook = QuantizationHooks(self.weight_quantizers[i], self.n_bit_weight)
            pre_hook = mod.register_forward_pre_hook(qhook.quantize_weights_hook)
            post_hook = mod.register_forward_hook(qhook.restore_weights_hook)
            self.hooks.append((pre_hook, post_hook, qhook))

    def update_measure(self, measure_in, measure_w):
        self.measure_w, self.measure_in = measure_w, measure_in
        for quantizer in self.input_quantizers:
            quantizer.measure = measure_in
        for quantizer in self.weight_quantizers:
            quantizer.measure = measure_w

    def update_nbits(self, n_bit_weight, n_bit_input):
        self.n_bit_weight = n_bit_weight
        self.n_bit_input = n_bit_input
        if USE_HOOKS:
            for _, _, qhook in self.hooks:
                qhook.n_bits = n_bit_weight

    

class QuantizationHooks:
    def __init__(self, w_quantizer, n_bits):
        self.w_quantizer = w_quantizer
        self.n_bits = n_bits

    def quantizer_w(self, weights):
        return self.w_quantizer(weights, num_bits=self.n_bits)

    def quantizer_b(self, bias):
        if self.w_quantizer.measure:
            return bias
        else:
            return quantize(bias, num_bits=self.n_bits, flatten_dims=(0, -1))
    
    def quantize_weights_hook(self, module, inputs):
        module._original_weights = module.weight.data.clone()
        module.weight.data = self.quantizer_w(module.weight.data)
        if module.bias is not None:
            module._original_biases = module.bias.data.clone()
            module.bias.data = self.quantizer_b(module.bias.data)

    def restore_weights_hook(self, module, inputs, outputs):
        module.weight.data = module._original_weights
        if module.bias is not None:
            module.bias.data = module._original_biases
        del module._original_weights
        if hasattr(module, '_original_biases'):
            del module._original_biases

def update_nbits(module, n_bit_weight=8, n_bit_input=8):
    for name, mod in module.named_children():
        if isinstance(mod, QuantizedLayer):
            mod.update_nbits(n_bit_weight, n_bit_input)
        else:
            update_nbits(mod, n_bit_weight, n_bit_input)



def set_measure(module, measure_in=False, measure_w=False, mods_measure=None):
    for name, mod in module.named_children():
        if isinstance(mod, QuantizedLayer):
            if mods_measure is not None:
                if mod.parent in mods_measure or mod.parent=="None":
                    mod.update_measure(measure_in, measure_w)
            else:
                mod.update_measure(measure_in, measure_w)
        set_measure(mod, measure_in, measure_w, mods_measure)



