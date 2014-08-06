require './lib/util'
require './lib/preprocessing'
require './cnn_model'
require './leave_one_subject_out_cv'

function main()
   local preprocessing = {
      preprocessing_lowpass_subject,
      preprocessing_lowpass_global
   }
   local sgd_config = {
      learningRate = 0.01,
      momentum = 0.9,
      xBatchSize = 15,
      xLearningRateDecay = 0.9
   }
   local cv = leave_one_subject_out_cv(cnn_model, preprocessing, sgd_config, 30)
   print(cv)
end
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
torch.manualSeed(13)
main()
