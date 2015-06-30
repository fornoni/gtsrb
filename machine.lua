require 'torch'
require 'xlua'
require "utils"


--p=xlua.Profiler();

--[[ Class containing a general training algorithm with its default parameters]]
local machine = torch.class('machine')

--[[ Initialization method, assigning the parameters for the machine a default value
ARGS:
- `trainData`            : training dataset object
- `model`                : model object
- `optAlgo`              : optimization algorithm object
]]
function machine:__init(trainData, model, logDir)

  --initializes the model
  model:getModel();


  self.maxEpochs= 10;
  self.batchSize=1;

  --if inputType is 'double' or 'cuda', the input will be converted in place to match the desired format
  self.inputType='';

  self.model=model;

  -- the total training time for the architecture will be stored here
  self.totTrainTime=0;
  -- the total training epochs for the architecture will be stored here
  self.trainedEpochs=0;

  print(self.model.net)
  --print(self.model.net:listModules())

  -- Creates a confusion matrix object
  self.confusion = optim.ConfusionMatrix(trainData.classNames);
  -- Retrieves parameters and gradients (these are tensors that will be automatically updated during training)
  self.parameters,self.gradParameters = model.net:getParameters();

  -- prints out some information about the network
  local tr_params = "# of trainable parameters: " .. (self.gradParameters:size())[1]
  print("\nCNN using architecture: " ..tostring(model).."")
  --  print(self.model.net)
  print(tr_params .."\n")

  --  self.opt_str="Training using: " .. tostring(self.model.optAlgo) .. ", with parameters: " .. self.model.optAlgo:parstostring "\n";
  self.opt_str="Training using: " .. tostring(self.model.optAlgo) .. " with:\n learning rate = "..self.model.optAlgo.learningRate .. "\n learning rate decay = " .. self.model.optAlgo.learningRateDecay .. "\n momentum = "..self.model.optAlgo.momentum .."\n weight decay = "..self.model.optAlgo.weightDecay .. "\n";


  -- id associated to the machine
  self.machineId = tostring(model) ..'_' .. tostring(self.model.optAlgo) .. '_BS' .. self.batchSize


  -- logs utils
  self.saveModel=true;
  self.logDir=logDir;
  self.modelDir= logDir .. '/' .. self.machineId .. "_models"

  if self.saveModel and not file_exists(self.modelDir) then
    os.execute("mkdir " .. self.modelDir)
  end

  self:initLogger()

  -- logs out some information about the network
  self.trLog:add{tostring(self.model.net).."\n"..tr_params .."\n" .. self.opt_str, '' , '' , '' }

  -- If necessary it optimizes the machine for running on cuda
  if self.inputType == 'cuda' then
    self.model.net:cuda()
    self.model.loss:cuda()
  end

end

--[[ Initializes the logger utility ]]
function machine:initLogger()
  local time = os.date("*t")
  time = time.month .. time.day .. time.hour .. time.min
  local tr_log_f = self.logDir .. '/' .. self.machineId .. '_' .. time .. '_train.log'
  local te_log_f = self.logDir .. '/' .. self.machineId .. '_' .. time .. '_test.log'
  self.trLog = optim.Logger(tr_log_f)
  self.teLog = optim.Logger(te_log_f)
  self.trLog:setNames{'arch', 'error rate', 'loss', 'train time'}
  self.teLog:setNames{'error rate', 'test time'}
end


function machine:getModelFile(epoch)
  local epoch = epoch or self.trainedEpochs;
  return self.modelDir .. "/" ..  self.model.optAlgo:parstostring() .. "_epoch" .. epoch .. ".net" ;
end

--[[saves the machine current status]]
function machine:save()
  print('Saving machine to '.. self:getModelFile())
  torch.save(self:getModelFile(), self)
end

--[[loads the machine in its last saved status]]
function machine:load()
  local modelFile;
  for ep=self.maxEpochs,1,-1 do
    modelFile=self:getModelFile(ep)
    --self.modelDir .. "/epoch" .. ep .. ".net" ;
    --    print("trying to load: " ..  modelFile)
    if file_exists(modelFile) then
      break
    end
  end
  print('Loading machine from '.. modelFile)
  return torch.load(modelFile)
end

--[[ Runs an experiment by performing a set of epochs of training, each followed by a testing
ARGS:
- `trainData`            : training dataset object
- `testData`             : testing dataset object
 ]]
function machine:runExp(trainData,testData,maxTrain)

  maxTrain= maxTrain or math.huge
  local maxEpochs=self.maxEpochs
  for epoch=self.trainedEpochs+1,maxEpochs do
    self.trainedEpochs=epoch

    torch.manualSeed(1e6+epoch*1e3+1)

    --    if epoch==10 then
    --      self.batchSize=self.trainData:size()
    --      self.model.optAlgo=lbfgs()
    --    end
    --    self.batchSize=self.batchSize*epoch

    self:train(trainData,maxTrain)
    self:test(testData)
  end

  print("Total training time =" .. self.totTrainTime .."s\n\n")

end

--[[ Trains the machine for one epoch
ARGS:
- `trainData`            : training dataset object
]]
function machine:train(trainData,maxTrain)

  print(self.opt_str)

  maxTrain= maxTrain or math.huge

  local time = sys.clock()
  local ntr=math.min(trainData:size(),maxTrain)
  local loss=0;

  -- sets model to training mode
  self.model.net:training()

  -- shuffles the samples and saves the results
  local shuffle = torch.randperm(ntr)

  -- trains for one epoch
  print(" epoch # " .. self.trainedEpochs .. '/' .. self.maxEpochs .. ' [batchSize = ' .. self.batchSize .. ']')
  for t = 1,ntr,self.batchSize do

    -- disp progress
    xlua.progress(torch.ceil(t/self.batchSize), torch.round(ntr/self.batchSize))

    -- creates a mini batch
    local inputs = {}
    local targets = {}

    for i = t,math.min(t+self.batchSize-1,ntr) do

      -- load new sample
      local input = trainData.data[shuffle[i]]
      local target = trainData.labels[shuffle[i]]

      --converts the sample to the desired format (to get the desired precision)
      if self.inputType == 'double' then input = input:double()
      elseif self.inputType == 'cuda' then input = input:cuda() end

      --adds the selected inputs and targets to the appropriate tables
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    --calls the optimization algorithm, passing to it the closure function using the current mini-batch, the current architecture parameters and the optimization params
    local slck,err = self.model.optAlgo.optimize(self:getFwdBwdFunc(inputs, targets), self.parameters, self.model.optAlgo)
    loss= loss + err[1]

  end

  -- computes 1 epoch training time and adds it to the total training time
  time = sys.clock() - time
  self.totTrainTime=self.totTrainTime+time;

  -- prints train error rate and loss
  self.confusion:updateValids()
  local err_rate = (1 - self.confusion.totalValid) * 100;
  print('\n train error rate = '..err_rate..'%')
  print(' loss = '..loss)

  -- prints time taken
  print(" training time = " .. time .. 's')
  print(" time / sample = " .. (time / ntr*1000) .. 'ms\n')

  -- logs out accuracy, loss and train time
  self.trLog:add{ '', err_rate, loss, time }

  -- next epoch
  self.confusion:zero()

  --if required saves the current model
  if self.saveModel then
    self:save()
  end
end

--[[ Tests the machine
ARGS:
- `testData`             : testing dataset object
]]
function machine:test(testData)
  -- local vars
  local time = sys.clock()
  local nte=testData:size();

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  self.model.net:evaluate()

  -- test over test data
  print('Testing:')
  for t = 1,nte do
    -- disp progress
    xlua.progress(t, nte)

    -- get new sample
    local input = testData.data[t]
    --converts the data sample to the desired format (to get the desired precision)
    if self.inputType == 'double' then input = input:double()
    elseif self.inputType == 'cuda' then input = input:cuda() end
    local target = testData.labels[t]

    -- test sample
    local pred = self.model.net:forward(input)
    self.confusion:add(pred, target)
  end

  -- computes testing time
  time = sys.clock() - time

  -- prints error rate
  self.confusion:updateValids()
  local err_rate = (1 - self.confusion.totalValid) * 100;
  print('\n test error rate = '..err_rate..'%')

  -- prints time taken
  print(" testing time = " .. time .. 's')
  print(" time / sample = " .. (time / nte*1000) .. 'ms\n')

  -- logs out accuracy
  self.teLog:add{err_rate, time}

  -- next iteration:
  self.confusion:zero()
end

--[[ Function used to evaluate the network (forward) on some inputs, compute the loss and the gradient vector (backward) accordingly
ARGS:
- `inputs`             : the training data samples to be used by the function
- `targets`            : the corrisponding labels]]
function machine:getFwdBwdFunc(inputs, targets)

  -- create closure to evaluate f(X) and df/dX
  local fwdBwdFunc=function(x)
    -- get new parameters
    if x ~= self.parameters then
      self.parameters:copy(x)
    end

    -- reset gradients
    self.gradParameters:zero()

    -- loss is the average of all losses
    local loss = 0

    -- evaluates function for complete mini batch
    for i = 1,#inputs do

      -- computes the loss for the sample
      local output = self.model.net:forward(inputs[i])
      local err = self.model.loss:forward(output, targets[i])
      loss = loss + err

      -- estimate the gradient df/dW
      local df_do = self.model.loss:backward(output, targets[i])
      self.model.net:backward(inputs[i], df_do)

      -- update confusion
      self.confusion:add(output, targets[i])
    end

    -- averages gradients and loss
    self.gradParameters:div(#inputs)
    loss = loss/#inputs

    --    xlua.print(loss)
    -- return f and df/dX
    return loss,self.gradParameters
  end

  return fwdBwdFunc
end


--[[ Tests the machine
ARGS:
- `testData`             : testing dataset object
- `n_layers`             : the number of layers to be removed from the network to use it as a feture extractor
]]
function machine:extractFeats(testData, n_layers)

  n_layers =  n_layers or self.model.nFeatsLayers
  
  if not self.model.featureExtractor then
    self.model:toFeatureExtractor(n_layers)
    self.model.featureExtractor=true
  end

  -- local vars
  local time = sys.clock()
  local nte=testData:size();

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  self.model.net:evaluate()

  -- gets number of output features
  local lm=self.model.net:listModules()
  local nf=lm[#lm].output:size()[1]

  -- create a tensor for the features
  local feats=torch.Tensor(nte,nf)

  -- test over test data
  print('Extracting features, using:')
  print(self.model.net)
  for t = 1,nte do
    -- disp progress
    xlua.progress(t, nte)

    -- get new sample
    local input = testData.data[t]

    --converts the data sample to the desired format (to get the desired precision)
    if self.inputType == 'double' then input = input:double()
    elseif self.inputType == 'cuda' then input = input:cuda() end

    -- extract the features sample
    feats[{{t},{}}] = self.model.net:forward(input)
  end

  -- computes testing time
  time = sys.clock() - time

  -- prints time taken
  print(" extraction time = " .. time .. 's')
  print(" time / sample = " .. (time / nte*1000) .. 'ms\n')

  return feats
end
