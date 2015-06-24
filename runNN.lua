require "dataset"
require "algo"
require "machine"
require "utils"

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-arch', 'mpNet', 'specifies which of the architectures to use')
cmd:option('-epochs', 100, 'the number of training epochs to run the architecture')
cmd:option('-data', '/home/datasets/GTSR2/data', 'specifies the directory where the data is')
cmd:option('-nl', 3, 'specifies the number of *non-convolutional* layers to be removed from the CNN to produce a feature extractor')
cmd:option('-orig', false, 'if true, the pre-processed image features are directly fed to the NN classifier')
cmd:text()
opt = cmd:parse(arg or {})

local function main()
  -- sets GTSR data folders
  local data_dir=opt.data
  local results_dir = data_dir .. '/results'

  -- gets the training and testing data
  local trainData,testData=geGTSRtData(data_dir)

  if opt.orig then
    print('\nReshaping data to be fed to the NN classifier')
    trainData.data=trainData.data:reshape(trainData:size(),trainData:dim()*trainData:dim())
    testData.data=testData.data:reshape(testData:size(),testData:dim()*testData:dim())
    print('\nTesting, using a Nearest Neighbor on pre-processed input images:')
    kNN(1,trainData,testData)
  else
    --selects which architecture to use
    print("Architecture requested: "..opt.arch)
    require("architectures." .. opt.arch)
    
    -- sets torch global parameters
    torch.manualSeed(1)
    torch.setnumthreads(2)
    torch.setdefaulttensortype('torch.FloatTensor')

    -- instantiates the specified convnet model
    local model=_G[opt.arch](trainData)
    local convnet=machine(trainData,model,results_dir)
    convnet.maxEpochs=opt.epochs
    convnet=convnet:load()
    convnet:initLogger()
    --removes the last nl layers
    trainData,testData = extracFeats(convnet, opt.nl, trainData, testData, data_dir)
    print('\nTesting, using a Nearest Neighbor approach on features extracted using the CNN')

    kNN(1,trainData,testData)
  end
end

main()
