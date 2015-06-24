require "dataset"
require "algo"
require "machine"
require "utils"

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-arch', 'mpNet', 'specifies which of the architectures to use')
cmd:option('-data', '/home/datasets/GTSR/data', 'specifies the directory where the data is')
cmd:text()
opt = cmd:parse(arg or {})

--selects which architecture to use
require("architectures." .. opt.arch)

local function main()
  -- sets GTSR data folders
  local data_dir=opt.data
  local log_dir = data_dir .. '/results'

  -- gets the training and testing data
  local trainData,testData=geGTSRtData(data_dir)

  -- sets torch global parameters
  torch.manualSeed(1)
  torch.setnumthreads(2)
  torch.setdefaulttensortype('torch.FloatTensor')
  
  -- instantiates a convnet model
  local model=mpNet(trainData)
  
  local convnet=machine(trainData,testData,model,log_dir)
  convnet.maxEpochs=100
  convnet:runExp()
end

main()
