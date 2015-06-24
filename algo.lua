require 'optim'


--[[ Class representing a sgd algorithm]]
local sgd = torch.class('sgd')
--sgd = {}
--[[ Initialization method, assigning the parameters for the algorithm a default value
ARGS:
- `learningRate`            : learning rate
- `weightDecay`             : weightDecay
- `momentum`                : momentum
- `learningRateDecay`       : learningRateDecay
]]
function sgd:__init(learningRate,weightDecay,momentum,learningRateDecay)
  self.learningRate = learningRate or 1e-3;
  self.weightDecay = weightDecay or 0;
  self.momentum = momentum or 0;
  self.learningRateDecay= learningRateDecay or 1e-7;
  self.optimize = optim.sgd;
end


--[[ Class representing a lbfgs algorithm]]
local lbfgs = torch.class('lbfgs')
--[[ Initialization method, assigning the parameters for the algorithm a default value
ARGS:
- `learningRate`            : learning rate
- `maxIter`                 : maxIter
- `nCorrection`             : nCorrection
]]
function lbfgs:__init(learningRate,maxIter,nCorrection)
  self.learningRate = learningRate or 1e-3;
  self.maxIter = maxIter or 2;
  self.nCorrection = nCorrection or 10;
  self.optimize = optim.lbfgs;
end
