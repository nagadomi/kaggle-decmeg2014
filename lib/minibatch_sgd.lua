require 'optim'
require 'xlua'

function minibatch_sgd(model, criterion, train_data, config)
   local opt = config or {}
   local batch_size = opt.xBatchSize or 15
   local shuffle = torch.randperm(#train_data)
   local parameters, grad_parameters = model:getParameters()
   local acc = 0
   local count = 0
   
   for i = 1, #train_data, batch_size do
      local inputs = {}
      local targets = {}

      if i + batch_size > #train_data then
	 break
      end
      local feval = function(x)
	 local loss = 0
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 grad_parameters:zero()
	 for j = i, i + batch_size do
	    local x = train_data[shuffle[j]][1]
	    local y = train_data[shuffle[j]][2]
	    local z = model:forward(x)
	    if (z[1][1] > 0.5 and y[1] == 1) or (z[1][1] <= 0.5 and y[1] ~= 1) then
	       acc = acc + 1
	    end
	    count = count + 1
	    loss = loss + criterion:forward(z, y)
	    model:backward(x, criterion:backward(z, y))
	 end
	 loss = loss / batch_size
	 grad_parameters:div(batch_size)
	 return loss, grad_parameters
      end
      xlua.progress(i, #train_data)
      optim.sgd(feval, parameters, config)
   end
   print("\n")
   return acc / count
end
