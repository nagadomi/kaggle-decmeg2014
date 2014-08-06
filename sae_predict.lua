require './lib/util'
require './lib/preprocessing'
require './sae_model'

local function add_submission(fp, model, test_data)
   for i = 1, #test_data do
      local y = model:forward(test_data[i][1])
      if y[1][1] > 0.5 then
	 fp:write(string.format("%d,%d\n", test_data[i][2], 0))
      else
	 fp:write(string.format("%d,%d\n", test_data[i][2], 1))
      end
   end
end

function main()
   local subjects = {17, 18, 19, 20, 21, 22, 23}
   local data = {}
   local test_data = {}
   local mean = torch.load("model/mean.bin")
   local std = torch.load("model/std.bin")
   
   for i = 1, #subjects do
      data[subjects[i]] = load_data(subjects[i], false)
      preprocessing_gauss_subject(data[subjects[i]])
      concat_table(test_data, data[subjects[i]])
      print("load .. ", subjects[i])
   end
   preprocessing_gauss_global(test_data, mean, std)

   local fp = io.open("./submission.txt", "w")
   fp:write("Id,Prediction\n")
   for i = 1, #subjects do
      local model = torch.load(string.format("model/sae_%d_epoch_10.model", subjects[i]))
      add_submission(fp, model, data[subjects[i]])
   end
   fp:close()
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
torch.manualSeed(13)
main()
