require 'nn'
require './lib/util'
require './lib/SplitSensorType'
require './lib/JoinSensorType'
require './lib/GradientBooster'

function sae_model(subject_id)
   local model = nn.Sequential()
   local parallel = nn.ParallelTable()
   
   -- seprating gradiometers and magnetometers
   model:add(nn.SplitSensorType())
   for i = 1, 2 do
      -- weight-sharing MLP layers
      local mlp = nil
      if i == 1 then
	 -- gradiometers
	 
	 -- 204 MLPs
	 mlp = nn.Sequential()
	 mlp:add(nn.Reshape(1, 204, TM))
	 mlp:add(nn.SpatialConvolution(1, 32, TM, 1))
	 mlp:add(nn.SoftSign())
	 mlp:add(nn.SpatialConvolution(32, 16, 1, 1))
	 mlp:add(nn.SoftSign())
	 
	 -- initializing with denosing autoencoders
	 local ae = torch.load(string.format("model/grad_ae%d.model", subject_id))
	 mlp:get(2).weight:copy(ae:get(2).weight)
	 mlp:get(2).bias:copy(ae:get(2).bias)
	 mlp:get(4).weight:copy(ae:get(4).weight)
	 mlp:get(4).bias:copy(ae:get(4).bias)
      else
	 -- magnetometers

	 -- 102 MLPs
	 mlp = nn.Sequential()
	 mlp:add(nn.Reshape(1, 102, TM))
	 mlp:add(nn.SpatialConvolution(1, 32, TM, 1))
	 mlp:add(nn.SoftSign())
	 mlp:add(nn.SpatialConvolution(32, 16, 1, 1))
	 mlp:add(nn.SoftSign())
	 
	 -- initializing with denosing autoencoders
	 local ae = torch.load(string.format("model/mag_ae%d.model", subject_id))
	 mlp:get(2).weight:copy(ae:get(2).weight)
	 mlp:get(2).bias:copy(ae:get(2).bias)
	 mlp:get(4).weight:copy(ae:get(4).weight)
	 mlp:get(4).bias:copy(ae:get(4).bias)
      end
      parallel:add(mlp)
   end
   model:add(parallel)
   -- joining MLP outputs (16x204 + 16x102)
   model:add(nn.JoinSensorType())
   -- logistic regression layer
   model:add(nn.Reshape(CH * 16, 1, 1))
   model:add(nn.SpatialConvolution(CH * 16, 1, 1, 1))
   model:add(nn.Sigmoid())
   model:add(nn.Reshape(1))
   
   return model
end
