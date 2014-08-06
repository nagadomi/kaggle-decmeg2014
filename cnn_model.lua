require 'nn'
require './lib/util'
require './lib/DropChannel'
require './lib/SpatialAveragePooling'

-- Convolutional Neural Network

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
