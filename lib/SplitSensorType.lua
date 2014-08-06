local SplitSensorType, parent = torch.class('nn.SplitSensorType', 'nn.Module')

function SplitSensorType:__init()
   parent.__init(self)
   self.grad = torch.Tensor()
   self.mag = torch.Tensor()
end

function SplitSensorType:updateOutput(input)
   local mi = 1
   local gi = 1
   local input_size = 1
   local input_dim = input:size()
   
   for i = 1, input_dim:size() do
      input_size = input_size * input_dim[i]
   end
   
   self.grad:resize(204, input_size / 306)
   self.mag:resize(102, input_size / 306)
   for i = 1, 306 do
      if i % 3 == 0 then
	 self.mag[mi]:copy(input[i])
	 mi = mi + 1
      else
	 self.grad[gi]:copy(input[i])
	 gi = gi + 1
      end
   end
   self.output = {self.grad, self.mag}
   return self.output
end 

function SplitSensorType:updateGradInput(input, gradOutput)
   local mi = 1
   local gi = 1
   self.gradInput:resizeAs(input)
   for i = 1, 306 do
      if i % 3 == 0 then
	 self.gradInput[i]:copy(gradOutput[2][mi])
	 mi = mi + 1
      else
	 self.gradInput[i]:copy(gradOutput[1][gi])
	 gi = gi + 1
      end
   end
   return self.gradInput
end
