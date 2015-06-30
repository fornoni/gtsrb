require "dataset"
require "algo"
require "machine"
require "utils"

local function main()

  -- reads the command line options --
  local cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('-arch', 'mpNet', 'specifies which of the architectures to use')
  cmd:option('-epochs', 20, 'the number of training epochs to run the architecture')
  cmd:option('-data', '/home/datasets/GTSR3', 'specifies the directory where the data is')
  cmd:option('-load', false, 'if true tries to load last convnet and continue the training from there')
  cmd:option('-lr', -1, 'if true tries to load last convnet and continue the training from there')
  cmd:option('-decay',  -1, 'if true tries to load last convnet and continue the training from there')
  cmd:text()
  local opt = cmd:parse(arg or {})

  --selects which architecture to use
  print("Architecture requested: "..opt.arch)
  require("architectures." .. opt.arch)


  -- sets GTSR data folders
  local data_dir=opt.data
  local results_dir = data_dir .. '/results'

  -- gets the training and testing data
  local trainData,testData=geGTSRtData(data_dir)

  -- sets torch global parameters
  torch.manualSeed(1)
  torch.setnumthreads(2)
  torch.setdefaulttensortype('torch.FloatTensor')

  -- Ronan's hack to rescale the weights in each layer, according to the number of parameters (unused for the moment)
  local hack=false  
  if hack then
    --it overloads the default learning rate and decay
    opt.lr=1
    opt.decay=1e-5
    local oldaccgradparameters = nn.Linear.accGradParameters
    function nn.Linear.accGradParameters(self, input, gradOutput, scale)
      oldaccgradparameters(self, input, gradOutput, scale/self.weight:size(2))
      --      oldaccgradparameters(self, input, gradOutput, scale*math.min(1,1/torch.norm(self:getParameters(),2,1)[1]))
    end

    local oldaccgradparameters = nn.SpatialConvolutionMM.accGradParameters
    function nn.SpatialConvolutionMM.accGradParameters(self, input, gradOutput, scale)
      oldaccgradparameters(self, input, gradOutput, scale/(self.kW*self.kH*self.nInputPlane))
      --      oldaccgradparameters(self, input, gradOutput, scale*math.min(1,1/torch.norm(self:getParameters(),2,1)[1]))
    end
  end

  -- instantiates the specified convnet model
  local model=_G[opt.arch](trainData)

  -- if necessary, overloads the default hyper-parameters of the method --
  if opt.lr~=-1 then
    model.optAlgo.learningRate = opt.lr
  end
  if opt.decay~=-1 then
    model.optAlgo.learningRateDecay = opt.decay
  end

  -- instantiate a new learning machine --
  local convnet=machine(trainData,model,results_dir)
  convnet.maxEpochs=opt.epochs
  
  -- if necessary loads the last saved model --
  if opt.load then
    convnet=convnet:load()
    convnet.maxEpochs=opt.epochs
    convnet:initLogger()
    print('\nRe-testing convnet')
    -- convnet:test(trainData)
    convnet:test(testData)
  end
  
  -- defines the maximum amount of training data to be used (default is all of it)
  local maxTrain=math.huge
  convnet:runExp(trainData,testData,  maxTrain)
end

main()
