require 'nn'
require 'lib/DropChannel'
require 'lib/util'

-- Linear Model
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
