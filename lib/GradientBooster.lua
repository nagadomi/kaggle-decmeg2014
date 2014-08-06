require 'nn'

local GradientBooster, Parent = torch.class('nn.GradientBooster', 'nn.Module')

function GradientBooster:__init(scale)
   Parent.__init(self)
   if scale == nil then
      self.scale = 1.0
   else
      self.scale = scale
   end
end

function GradientBooster:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   return self.output
end

function GradientBooster:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput):mul(self.scale)
   return self.gradInput
end
