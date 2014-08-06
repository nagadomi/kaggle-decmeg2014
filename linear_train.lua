require 'lib/util'
require 'lib/preprocessing'
require 'lib/minibatch_sgd'
require './linear_model'

local function train(model_factory,
		     preprocessing,
		     sgd_config,
		     max_epoch)
   local subjects = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
   local test_subjects = {17,18,19,20,21,22,23}
   local learning_rate = sgd_config.learningRate or 0.01
   local learning_rate_decay = sgd_config.xLearningRateDecay or 1.0
   local train_data = {}
   local all = {}
   local acc = 0
   local cv = {}
   local criterion = nn.MSECriterion()
   local model = model_factory(subjects[i])
   
   for k,i in pairs(subjects) do
      print("load .. " .. i)
      local data = load_data(i, true)
      preprocessing[1](data)
      concat_table(all, data)
      concat_table(train_data, data)
   end
   for k,i in pairs(test_subjects) do
      print("load .. " .. i)
      local td = load_data(i, false)
      preprocessing[1](td)
      concat_table(all, td)
   end
   local mean = calc_mean(all)
   local std = calc_std(all, mean)
   torch.save("model/mean.bin", mean)
   torch.save("model/std.bin", std)
   
   preprocessing[2](all, mean, std)
   collectgarbage()
   
   print(model)
   for j = 1, max_epoch do
      print("# " .. j)
      sgd_config.learningRate = learning_rate * torch.pow(learning_rate_decay, j - 1)
      model:training()
      print(minibatch_sgd(model, criterion, train_data, sgd_config))
      model:evaluate()
      torch.save(string.format("model/linear_epoch_%d.model", j), model)
      collectgarbage()
   end
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(13)

local preprocessing = {
   preprocessing_lowpass_subject,
   preprocessing_lowpass_global
}
local sgd_config = {
   learningRate = 0.0001,
   momentum = 0.9,
   xbatchSize = 4,
   xLearningRateDecay = 0.9
}
train(linear_model, preprocessing, sgd_config, 15)
