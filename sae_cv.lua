require './lib/util'
require './lib/preprocessing'
require './sae_model'
require './leave_one_subject_out_cv'

function main()
   local preprocessing = {
      preprocessing_gauss_subject,
      preprocessing_gauss_global
   }
   local sgd_config = {
      learningRate = 0.01,
      momentum = 0.8,
      xBatchSize = 15,
      xLearningRateDecay = 0.9
   }
   local cv = leave_one_subject_out_cv(sae_model, preprocessing, sgd_config, 10)
   print(cv)
end
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
torch.manualSeed(13)
main()
