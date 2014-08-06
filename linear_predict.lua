require './lib/util'
require './lib/preprocessing'
require './linear_model'

function main()
   local subjects = {17, 18, 19, 20, 21, 22, 23}
   local data = {}
   local test_data = {}
   local mean = torch.load("model/mean.bin")
   local std = torch.load("model/std.bin")
   local model = torch.load("model/linear_epoch_15.model")
   print(model)
   for i = 1, #subjects do
      local data = load_data(subjects[i], false)
      preprocessing_lowpass_subject(data)
      concat_table(test_data, data)
      print("load .. ", subjects[i])
   end
   preprocessing_lowpass_global(test_data, mean, std)
   make_submission("./submission.txt", model, test_data)
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(13)
main()
