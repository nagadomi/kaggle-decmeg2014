require './lib/util'
require './lib/preprocessing'
require './linear_model'
require './leave_one_subject_out_cv'

function main()
   local preprocessing = {
      preprocessing_lowpass_subject,
      preprocessing_lowpass_global
   }
   local sgd_config = {
      learningRate = 0.0001,
      momentum = 0.9,
      xBatchSize = 4,
      xLearningRateDecay = 0.9
   }
   local cv = leave_one_subject_out_cv(linear_model, preprocessing, sgd_config, 15)
   print(cv)
end
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(13)
main()
