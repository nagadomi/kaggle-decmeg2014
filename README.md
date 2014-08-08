# kaggle-decmeg2014

code for DecMeg2014 competition.

http://www.kaggle.com/c/decoding-the-human-brain

# Requirements

- Ubuntu 14.04
- LuaJit/Torch7
- 32GB RAM

# Installation

Install Torch7:

    curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash

Install(or update) dependency packages:

    apt-get install libmatio2 libfftw3
    luarocks install torch
    luarocks install nn
    luarocks install unsup
    luarocks install signal
    luarocks install https://raw.githubusercontent.com/soumith/matio-ffi.torch/master/matio-scm-1.rockspec

Convert dataset:

Place the data files into a subfolder ./data/ first.

    th convert_data.lua

# Models

## Linear

Logistic Regression with Dropout Regularization.

```lua
function linear_model()
   local model = nn.Sequential()   
   -- randomly drop sensors
   model:add(nn.DropChannel(0.1))
   -- logistic regression layer
   model:add(nn.Reshape(CH * TM))
   model:add(nn.Linear(CH * TM, 1))
   model:add(nn.Sigmoid())
   model:add(nn.Reshape(1))
   return model
end
```

### Running the leave-one-subject-out CV

    th linear_cv.lua

### Generating the submission.txt

    th linear_train.lua
    th linear_predict.lua

## Convolutional Neural Networks

```lua
local function frontal_occipital_connection(n)
   -- separating occipital lobe and frontal lobe
   local SEPARATION = 163
   local ft = {}
   local c = 1
   for j = 1, n do
      for i = 1, CH do
	 if i < SEPARATION then
	    ft[c] = {}
	    ft[c][1] = i
	    ft[c][2] = j
	 else
	    ft[c] = {}
	    ft[c][1] = i
	    ft[c][2] = n + j
	 end
	 c = c + 1 
      end
   end
   return torch.Tensor(ft)
end
function cnn_model()
   local model = nn.Sequential()
   -- randomly drop sensors
   model:add(nn.DropChannel(0.4))
   -- convolution layers
   model:add(nn.Reshape(CH, 1, TM))
   model:add(nn.SpatialConvolutionMap(frontal_occipital_connection(32), 9, 1, 1))
   model:add(nn.SoftSign())
   model:add(nn.SpatialAveragePooling(64, 2, 1, 2, 1))
   model:add(nn.SpatialConvolution(64, 32, 8, 1, 1, 1))
   model:add(nn.SoftSign())
   model:add(nn.SpatialAveragePooling(32, 2, 1, 2, 1))
   model:add(nn.SpatialConvolution(32, 16, 9, 1, 1, 1))
   model:add(nn.SoftSign())
   model:add(nn.SpatialAveragePooling(16, 2, 1, 2, 1))
   -- fully connected layer
   model:add(nn.SpatialConvolution(16, 32, 12, 1))
   model:add(nn.SoftSign())
   model:add(nn.Dropout(0.5))
   model:add(nn.SpatialConvolution(32, 1, 1, 1))
   model:add(nn.Sigmoid())
   model:add(nn.Reshape(1))
   return model
end
```

key parts:
- Applying the stronger dropout at input layer.
- SoftSign + Average Pooling is better than ReLU + Max Pooling in this data.

### Running the leave-one-subject-out CV

    th cnn_cv.lua

### Generating the submission.txt

    th cnn_train.lua
    th cnn_predict.lua

## Stacked Autoencoders

In the first plan, this model is based on Stacked Denosing Autoencoders.
In the final submission, no denosing, no stacking...

```lua
function sae_model(subject_id)
   local model = nn.Sequential()
   local parallel = nn.ParallelTable()
   
   -- seprating gradiometers and magnetometers
   model:add(nn.SplitSensorType())
   for i = 1, 2 do
      -- MLP layer
      local mlp = nil
      if i == 1 then
	 -- gradiometers
	 
	 -- 204 MLPs (weight-sharing)
	 mlp = nn.Sequential()
	 mlp:add(nn.Reshape(1, 204, TM))
	 mlp:add(nn.SpatialConvolution(1, 32, TM, 1))
	 mlp:add(nn.SoftSign())
	 mlp:add(nn.SpatialConvolution(32, 16, 1, 1))
	 mlp:add(nn.SoftSign())
	 
	 -- initializing with autoencoders
	 local ae = torch.load(string.format("model/grad_ae%d.model", subject_id))
	 mlp:get(2).weight:copy(ae:get(2).weight)
	 mlp:get(2).bias:copy(ae:get(2).bias)
	 mlp:get(4).weight:copy(ae:get(4).weight)
	 mlp:get(4).bias:copy(ae:get(4).bias)
      else
	 -- magnetometers

	 -- 102 MLPs (weight-sharing)
	 mlp = nn.Sequential()
	 mlp:add(nn.Reshape(1, 102, TM))
	 mlp:add(nn.SpatialConvolution(1, 32, TM, 1))
	 mlp:add(nn.SoftSign())
	 mlp:add(nn.SpatialConvolution(32, 16, 1, 1))
	 mlp:add(nn.SoftSign())
	 
	 -- initializing with autoencoders
	 local ae = torch.load(string.format("model/mag_ae%d.model", subject_id))
	 mlp:get(2).weight:copy(ae:get(2).weight)
	 mlp:get(2).bias:copy(ae:get(2).bias)
	 mlp:get(4).weight:copy(ae:get(4).weight)
	 mlp:get(4).bias:copy(ae:get(4).bias)
      end
      parallel:add(mlp)
   end
   model:add(parallel)
   -- joining MLP outputs (204x16 + 102x16)
   model:add(nn.JoinSensorType())
   -- logistic regression layer
   model:add(nn.Reshape(CH * 16, 1, 1))
   model:add(nn.SpatialConvolution(CH * 16, 1, 1, 1))
   model:add(nn.Sigmoid())
   model:add(nn.Reshape(1))
   
   return model
end
```

### Running the leave-one-subject-out CV

    th sae_pretrain.lua
    th sae_cv.lua
    
### Generating the submission.txt

    th sae_pretrain.lua
    th sae_train.lua
    th sae_predict.lua

# Figure

score

| Subject | LR | CNN   | SAE    |
| :-- | :----- | :----- | :----- |
| 01 | 0.7811 | 0.7912 | 0.7929 |
| 02 | 0.7337 | 0.6979 | 0.7030 |
| 03 | 0.6487 | 0.6453 | 0.6470 |
| 04 | 0.7811 | 0.8215 | 0.8215 |
| 05 | 0.6877 | 0.7354 | 0.6911 |
| 06 | 0.6224 | 0.6802 | 0.6853 |
| 07 | 0.7278 | 0.7482 | 0.7278 |
| 08 | 0.7010 | 0.7060 | 0.7145 |
| 09 | 0.7373 | 0.7693 | 0.7390 |
| 10 | 0.7016 | 0.7203 | 0.6915 |
| 11 | 0.7246 | 0.7280 | 0.7195 |
| 12 | 0.7320 | 0.7474 | 0.7593 |
| 13 | 0.7006 | 0.6955 | 0.6853 |
| 14 | 0.7363 | 0.7517 | 0.7448 |
| 15 | 0.7068 | 0.6948 | 0.7120 |
| 16 | 0.5796 | 0.5796 | 0.5864 |
| CV mean (1-16) | 0.7064 | 0.7195| 0.7138 |
| public LB (17-19) | 0.69615 | 0.6967 | 0.7080 |
| private LB (20-23) | 0.6861 | 0.6804 | 0.6900 |


NOTICE: CNN/SAE is unstable.(+-0.01)

## References

- Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov,  Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
- P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio,  and P. Manzagol,  Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion.
- [MEG概説](http://www.nips.ac.jp/~nagata/MEG/MEGoutline.pdf)
