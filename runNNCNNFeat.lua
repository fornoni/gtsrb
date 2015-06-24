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
  convnet=convnet:load()
  convnet:initLogger()

  --  print('\nTesting, using a Nearest Neighbor on input images:')
  --  NN(trainData,testData)

  trainData,testData = extracFeats(convnet, trainData, testData,data_dir)

  print('\nTesting, using a Nearest Neighbor approach on features extracted using the CNN:')
  NN(trainData,testData)

end

main()
