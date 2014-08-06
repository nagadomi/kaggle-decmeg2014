local JoinSensorType, parent = torch.class('nn.JoinSensorType', 'nn.Module')

function JoinSensorType:__init()
   parent.__init(self)
   self.grad_tensor = torch.Tensor()
   self.mag_tensor = torch.Tensor()
end

function JoinSensorType:updateOutput(input)
   local grad_size = input[1]:size() -- x, 204
   local mag_size = input[2]:size()  -- x, 102
   
   self.grad_size = 1
   for i = 1, grad_size:size() do
      self.grad_size = self.grad_size * grad_size[i]
   end
   self.mag_size = 1
   for i = 1, mag_size:size() do
      self.mag_size = self.mag_size * mag_size[i]
   end
   self.grad_size = self.grad_size / 204
   self.mag_size = self.mag_size / 102
   
   local grad_output = input[1]:clone():resize(self.grad_size, 204):t()
   local mag_output = input[2]:clone():resize(self.mag_size, 102):t()
   self.output:resize(102, self.grad_size * 2 + self.mag_size)
   for i = 1, 102 do
      self.output[i]:narrow(1, 1, self.grad_size):copy(grad_output[(i - 1) * 2 + 1])
      self.output[i]:narrow(1, self.grad_size + 1, self.grad_size):copy(grad_output[(i - 1) * 2 + 2])
      self.output[i]:narrow(1, self.grad_size * 2 + 1, self.mag_size):copy(mag_output[i])
   end
   return self.output
end 

function JoinSensorType:updateGradInput(input, gradOutput)
   local grad_output = torch.Tensor(204, self.grad_size)
   local mag_output = torch.Tensor(102, self.mag_size)
   for i = 1, 102 do
      grad_output[(i - 1) * 2 + 1]:copy(gradOutput[i]:narrow(1, 1, self.grad_size))
      grad_output[(i - 1) * 2 + 2]:copy(gradOutput[i]:narrow(1, self.grad_size + 1, self.grad_size))
      mag_output[i]:copy(gradOutput[i]:narrow(1, self.grad_size * 2 + 1, self.mag_size))
   end
   self.gradInput = { self.grad_tensor, self.mag_tensor }
   self.gradInput[1]:resizeAs(input[1]):copy(grad_output:t())
   self.gradInput[2]:resizeAs(input[2]):copy(mag_output:t())
   return self.gradInput
end
