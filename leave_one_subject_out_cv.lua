require 'lib/util'
require 'lib/minibatch_sgd'
require 'gnuplot'

function leave_one_subject_out_cv(model_factory,
				  preprocessing,
				  sgd_config,
				  max_epoch)
   local subjects = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
   local test_subjects = {17,18,19,20,21,22,23}
   local learning_rate = sgd_config.learningRate or 0.01
   local learning_rate_decay = sgd_config.xLearningRateDecay or 1.0
   local data = {}
   local all = {}
   local acc = 0
   local cv = {}
   
   for k,i in pairs(subjects) do
      print("load .. " .. i)
      data[i] = load_data(i, true)
      preprocessing[1](data[i])
      concat_table(all, data[i])
   end
   for k,i in pairs(test_subjects) do
      print("load .. " .. i)
      local td = load_data(i, false)
      preprocessing[1](td)
      concat_table(all, td)
   end
   local mean = calc_mean(all)
   local std = calc_std(all, mean)
   preprocessing[2](all, mean, std)
   all = nil
   
   for i = 1, #subjects do
      local epoch = 1
      local train_data = {}
      local criterion = nn.MSECriterion()
      local test_data = data[i]
      local train_acc = torch.Tensor(max_epoch):zero()
      local test_acc = torch.Tensor(max_epoch):zero()
      local model = model_factory(subjects[i])
      local sgd_config_local = {}

      for k, v in pairs(sgd_config) do
	 sgd_config_local[k] = v
      end
      
      print("### subject " .. subjects[i])
      for j = 1, #subjects do
	 if subjects[i] ~= subjects[j] then
	    concat_table(train_data, data[subjects[j]])
	 end
      end
      if i == 1 then
	 print(model)
      end
      for j = 1, max_epoch do
	 print("# " .. epoch)
	 -- learning rate decay
	 sgd_config_local.learningRate = learning_rate * torch.pow(learning_rate_decay, j - 1)
	 -- training the model
	 model:training()
	 train_acc[j] = minibatch_sgd(model, criterion, train_data, sgd_config_local)
	 model:evaluate()
	 -- validation
	 test_acc[j] = validate(model, test_data)
	 epoch = epoch + 1
	 print(train_acc[j])
	 print(test_acc[j])
	 --[[
	 gnuplot.plot(
	    {'training accuracy', train_acc:narrow(1, 1, j), '-'},
	    {'test accuracy', test_acc:narrow(1, 1, j), '-'}
	 )
	 --]]
	 collectgarbage()
      end
      print("@@ CV " .. subjects[i] )
      cv[subjects[i]] = validate(model, test_data)
      acc = acc + cv[subjects[i]]
      print(cv[subjects[i]])
   end
   print("\nmean accuracy = " .. (acc/ #subjects))
   
   return cv
end
