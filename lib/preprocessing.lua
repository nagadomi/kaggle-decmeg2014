require 'torch'
require 'image'
local signal = require 'signal'
require './util'

-- lowpass filter version

function preprocessing_lowpass_subject(x)
   for i = 1, #x do
      -- apply the lowpass filter
      lowpass_filter(x[i][1], 25, 25)
      -- select 100-250 timeseries for each channels
      x[i][1] = x[i][1]:narrow(2, TM_BEGIN, TM):clone()
   end
   local mean = calc_mean(x)
   local std = calc_std(x, mean)
   for i = 1, #x do
      -- standardize (aka z-score)
      x[i][1]:add(-1.0, mean)
      x[i][1]:cdiv(std)
      -- cut off outlier
      clipping(x[i][1], 2.0, -2.0)
      -- revert to raw space
      x[i][1]:cmul(std)
   end
   -- subtracting the mean
   mean = calc_mean(x)
   for i = 1, #x do
      x[i][1]:add(-1.0, mean)
   end
   collectgarbage("collect")
   
   return x
end
function preprocessing_lowpass_global(x, mean, std)
   local i
   for i = 1, #x do
      -- standardize
      x[i][1]:add(-mean)
      x[i][1]:cdiv(std)
      -- cut off outlier
      clipping(x[i][1], 2.0, -2.0)
   end
end

-- gaussian filter version

function preprocessing_gauss_subject(x)
   for i = 1, #x do
      x[i][1] = x[i][1]:narrow(2, TM_BEGIN, TM):clone()
   end
   local mean = calc_mean(x)
   local std = calc_std(x, mean)
   for i = 1, #x do
      x[i][1]:add(-1.0, mean)
      x[i][1]:cdiv(std)
      clipping(x[i][1], 2.0, -2.0)
      x[i][1]:cmul(std)
      gaussian_filter(x[i][1])
      l1_normalization(x[i][1])
   end
   mean = calc_mean(x)
   for i = 1, #x do
      x[i][1]:add(-1.0, mean)
   end
   collectgarbage("collect")
   
   return x
end
function preprocessing_gauss_global(x, mean, std)
   local i
   for i = 1, #x do
      x[i][1]:add(-mean)
      x[i][1]:cdiv(std)
      clipping(x[i][1], 2.0, -2.0)
      gaussian_filter(x[i][1])
   end
end

function calc_mean(x)
   local mean = torch.Tensor(x[1][1]:size()):zero()
   local scale = 1.0 / #x
   for i = 1, #x do
      mean:add(x[i][1] * scale)
   end
   return mean
end
function calc_std(x, mean)
   local std = torch.Tensor(x[1][1]:size()):zero()
   local scale = 1.0 / (#x - 1)
   for i = 1, #x do
      local v = (mean - x[i][1])
      v:pow(v, 2.0)
      std:add(v * scale)
   end
   std:pow(0.5)
   return std
end
function clipping(x, max, min)
   -- if x[i] > max then x[i] = max
   x[torch.gt(x, max)] = max
   -- if x[i] < min then x[i] = min
   x[torch.lt(x, min)] = min
   return x
end
function gaussian_filter(x)
   -- apply the 1x5 gaussian filter
   local GAUSSIAN_KERNEL_SIZE = 5
   local kernel = image.gaussian1D(GAUSSIAN_KERNEL_SIZE):resize(1, GAUSSIAN_KERNEL_SIZE)
   kernel:div(kernel:norm(1))
   local sma = torch.conv2(x, kernel, 'F')
   x:copy(sma:narrow(2, 3, x:size(2)))
   return x
end
function l1_normalization(x)
   for i = 1, CH do
      x[i]:div(x[i]:norm(1))
   end
   return x
end
function lowpass_filter(x, hz1, hz2)
   -- apply the lowpass filter
   for j = 1, CH do
      local fft = signal.fft(x[j])
      if j % 3 == 0 then
	 fft:narrow(1, hz2 + 1, (x[j]:size(1) - hz2 + 1) - hz2 + 1):zero()
      else
	 fft:narrow(1, hz1 + 1, (x[j]:size(1) - hz1 + 1) - hz1 + 1):zero()
      end
      x[j]:copy(signal.ifft(fft):t()[1])
   end
end
