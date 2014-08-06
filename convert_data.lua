local matio = require('matio')

function load_mat(file, is_test)
   local CH = 306
   local TM = 375
   local x = matio.load(file, "X")
   local y
   local data = {}
   
   -- scaling
   x:mul(1.0e+12)
   
   if is_test then
      local id = matio.load(file, "Id")
      for i = 1, x:size(1) do
	 table.insert(data, {x[i]:float(), id[i][1]})
      end
   else
      local y = matio.load(file, "y")
      for i = 1, x:size(1) do
	 table.insert(data, {x[i]:float(), y[i][1] + 1})
      end
   end
   return data
end
function convert_train()
   for i = 1, 16 do
      print(string.format("convert .. %d", i))
      local x = load_mat(string.format("./data/train_subject%02d.mat", i))
      torch.save(string.format("./data/subject%02d.bin", i), x, "binary")
      collectgarbage("collect")
   end
end
function convert_test()
   for i = 17, 23 do
      print(string.format("convert .. %d", i))
      local x = load_mat(string.format("./data/test_subject%02d.mat", i), true)
      torch.save(string.format("./data/subject%02d.bin", i), x, "binary")
      collectgarbage("collect")
   end
end
torch.setdefaulttensortype('torch.FloatTensor')
convert_train()
convert_test()
