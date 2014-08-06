local DropChannel, Parent = torch.class('nn.DropChannel', 'nn.Module')

function DropChannel:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   if self.p >= 1 or self.p < 0 then
      error('<DropChannel> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function DropChannel:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   
   if self.train then
      self.noise:resize(input:size(1)):bernoulli(self.p)
      for i = 1, input:size(1) do -- each channels
	 if self.noise[i] > 0 then
	    self.output[i]:zero()
	 end
      end
   else
      self.output:mul(1-self.p)
   end
   return self.output
end

function DropChannel:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      for i = 1, input:size(1) do
	 if self.noise[i] > 0 then
	    self.gradInput[i]:zero()
	 end
      end
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end
