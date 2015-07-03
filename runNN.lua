require "dataset"
require "algo"
require "machine"
require "utils"

local function main()

  -- reads the command line options --
  local cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('-arch', 'mpNet', 'specifies which of the architectures to use')
  cmd:option('-epochs', 100, 'the target number of training epochs the architecture has been run')
  cmd:option('-data', '/home/datasets/GTSR3', 'specifies the directory where the data is')
  cmd:option('-nl', -1, 'specifies the number of *non-convolutional* layers to be removed from the CNN to produce a feature extractor')
  cmd:option('-k', 1, 'the number of nearest neighbors to be considered for classification')
  cmd:option('-bs', 1000, 'the size of the batch of samples classified at once by the kNN algorithm')
  cmd:option('-orig', false, 'if true, the pre-processed image features are directly fed to the NN classifier')
  cmd:text()
  local opt = cmd:parse(arg or {})

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
    kNN(opt.k,trainData,testData,opt.bs)
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

    -- instantiates the machine and resets the number of training epochs
    local convnet=machine(trainData,model,results_dir)
    convnet.maxEpochs=opt.epochs
    
    -- tries to reload the model and re-initialize the logger
    local convnet=convnet:load()
    convnet:initLogger()
    
    -- if necessary sets the number of layers to remove
    local n_layers=nil
    if opt.nl~=-1 then
      n_layers=opt.nl
    end

    -- uses the convnet to extract features from the training and testing data
    trainData,testData = extracFeats(convnet, trainData, testData, data_dir,n_layers)

    print('\nTesting, using a Nearest Neighbor approach on features extracted using the CNN')
    kNN(opt.k,trainData,testData,opt.bs)
  end
end

main()
