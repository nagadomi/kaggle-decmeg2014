require 'torch'
CH = 306
TM_ALL = 375
TM_BEGIN = 100
TM = 150

function load_data(subject_id, label_binalize)
   local data = {}
   local lb = label_binalize or false
   local x = torch.load(string.format("data/subject%02d.bin", subject_id))
   for i = 1, #x do
      if lb then
	 if x[i][2] == 1 then
	    x[i][2] = torch.Tensor(1):fill(1.0)
	 else
	    x[i][2] = torch.Tensor(1):fill(0.0)
	 end
      end
      x[i][3] = subject_id
   end
   return x
end
function validate(model, test_data)
   local acc = 0.0
   for i = 1, #test_data do
      local preds = model:forward(test_data[i][1])
      if preds[1][1] > 0.5 and test_data[i][2][1] == 1 then
	 acc = acc + 1
      end
      if preds[1][1] <= 0.5 and test_data[i][2][1] ~= 1 then
	 acc = acc + 1
      end
   end
   acc = acc / #test_data
   return acc
end
function shuffle(data)
   local indexes = torch.randperm(#data)
   local tmp = {}
   local i
   for i = 1, #data do
      tmp[i] = data[indexes[i]]
   end
   for i = 1, #data do
      data[i] = tmp[i]
   end
end
function concat_table(x, a)
   for i = 1, #a do
      table.insert(x, a[i])
   end
end
function make_submission(file, model, test_data)
   local fp = io.open(file, "w")
   fp:write("Id,Prediction\n")
   for i = 1, #test_data do
      local y = model:forward(test_data[i][1])
      if y[1][1] > 0.5 then
	 fp:write(string.format("%d,%d\n", test_data[i][2], 0))
      else
	 fp:write(string.format("%d,%d\n", test_data[i][2], 1))
      end
   end
   fp:close()
end
