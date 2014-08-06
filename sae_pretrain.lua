require 'torch'
require 'optim'
require 'nn'
require 'unsup'
require 'gnuplot'
require 'xlua'
require './lib/util'
require './lib/preprocessing'

function mag_data(x)
   local mag = {}
   local c = 1
   for i = 1, #x do
      for j = 1, CH do
	 if j % 3 == 0 then
	    table.insert(mag, x[i][1][j])
	 end
      end
   end
   return mag
end
function grad_data(x)
   local grad = {}
   local c = 1
   for i = 1, #x do
      for j = 1, CH do
	 if j % 3 ~= 0 then
	    table.insert(grad, x[i][1][j])
	 end
      end
   end
   return grad
end
function encode(encoder, x)
   local xx = {}
   for i = 1, #x do
      xx[i] = encoder:forward(x[i]):clone()
   end
   return xx
end
function train_unsup_minibatch(model, batch_size, dataset, epoch, config)
   local x, dl_dx = model:getParameters()
   local shuffle = torch.randperm(#dataset)
   local err = 0
   local c = 1
   for t = 1, #dataset, batch_size do
      xlua.progress(t, #dataset)
      c = c + 1
      if #dataset < t + batch_size then
	 break
      end
      local inputs = {}
      for i = t, math.min(t + batch_size - 1, #dataset) do
         local input = dataset[shuffle[i]]
         table.insert(inputs, input)
      end
      local feval = function()
	 local f = 0
	 dl_dx:zero()

	 for i = 1, #inputs do
	    f = f + model:updateOutput(inputs[i], inputs[i])
	    model:updateGradInput(inputs[i], inputs[i])
	    model:accGradParameters(inputs[i], inputs[i])
	 end
	 dl_dx:div(batch_size)
	 f = f / batch_size

	 err = err + f
	 return f,dl_dx
      end
      optim.sgd(feval, x, config)
   end
   print("\n #" .. epoch .. " error = " .. (err / #dataset))
end
function train_autoencoders(x, max_epoch, filename)
   local encoder, decoder, model
   local config = {learningRate = 1.0e-3, momentum = 0.8 }
   local epoch = 1
   local beta = 1.0
   local input = 1
   
   for i = 1, x[1]:dim() do
      input = input * x[1]:size(i)
   end
   -- autoencoders
   encoder = nn.Sequential()
   encoder:add(nn.Reshape(1, 1, TM))
   encoder:add(nn.SpatialConvolution(1, 32, input, 1))
   encoder:add(nn.SoftSign())
   encoder:add(nn.SpatialConvolution(32, 16, 1, 1))
   encoder:add(nn.SoftSign())
   decoder = nn.Sequential()
   decoder:add(nn.Reshape(16))
   decoder:add(nn.Linear(16, input))
   
   model = unsup.AutoEncoder(encoder, decoder, beta)
   for i = 1, max_epoch do
      encoder:training()
      train_unsup_minibatch(model, 30, x, epoch, config)
      encoder:evaluate()
      epoch = epoch + 1
      torch.save(filename, encoder)
   end
   return encoder, decoder
end
function main()
   local max_epoch = 30
   local subjects = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
   local data = {}
   local all = {}
   for i = 1, #subjects do
      data[subjects[i]] = load_data(subjects[i], false)
      preprocessing_gauss_subject(data[subjects[i]])
      concat_table(all,data[subjects[i]]) 
      print("load " .. i)
   end
   local mean = calc_mean(all)
   local std = calc_std(all, mean)
   
   torch.save("model/mean.bin", mean)
   torch.save("model/std.bin", std)
   
   preprocessing_gauss_global(all, mean, std)
   collectgarbage()
   
   for i = 1, #subjects do
      local target_data = data[subjects[i]]
      local target = grad_data(target_data)
      local grad_enc = string.format("model/grad_ae%d.model", subjects[i])
      local mag_enc = string.format("model/mag_ae%d.model", subjects[i])
      local encoder, decoder
      
      encoder, decoder = train_autoencoders(target, max_epoch, grad_enc)
      collectgarbage()
      
      target = mag_data(target_data)
      encoder, decoder = train_autoencoders(target, max_epoch, mag_enc)
      collectgarbage()
   end
end
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
main()
