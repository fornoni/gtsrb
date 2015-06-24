require "dataset"
require "algo"
require "machine"
require "utils"

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-arch', 'mpNet', 'specifies which of the architectures to use')
cmd:option('-epochs', 20, 'the number of training epochs to run the architecture')
cmd:option('-data', '/home/datasets/GTSR2/data', 'specifies the directory where the data is')
cmd:option('-load', false, 'if true tries to load last convnet and continue the training from there')
cmd:text()
opt = cmd:parse(arg or {})

--selects which architecture to use
print("Architecture requested: "..opt.arch)
require("architectures." .. opt.arch)

local function main()
  -- sets GTSR data folders
  local data_dir=opt.data
  local results_dir = data_dir .. '/results'

  -- gets the training and testing data
  local trainData,testData=geGTSRtData(data_dir)

  -- sets torch global parameters
  torch.manualSeed(1)
  torch.setnumthreads(2)
  torch.setdefaulttensortype('torch.FloatTensor')
  
  -- instantiates the specified convnet model
  local model=_G[opt.arch](trainData)
  
  local convnet=machine(trainData,model,results_dir)
  convnet.maxEpochs=opt.epochs
  if opt.load then
    convnet=convnet:load()
    convnet.maxEpochs=opt.epochs
    convnet:initLogger()
    print('\nRe-testing convnet')
    convnet:test(testData)
  end
  convnet:runExp(trainData,testData)
end

main()
