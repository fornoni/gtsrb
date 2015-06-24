require 'torch'
require 'xlua'


--p=xlua.Profiler();

--[[ Class containing a general training algorithm with its default parameters]]
local machine = torch.class('machine')

--[[ Initialization method, assigning the parameters for the machine a default value
ARGS:
- `trainData`            : training dataset object
- `testData`             : testing dataset object
- `model`                : model object
- `optAlgo`              : optimization algorithm object
]]
function machine:__init(trainData,testData, model, logDir, optAlgo)

  --initializes the model
  model:getModel();

  self.optAlgo=optAlgo or sgd();
  self.maxEpochs= 10;
  self.batchSize=1;

  --if inputType is 'double' or 'cuda', the input will be converted in place to match the desired format
  self.inputType='';

  self.model=model;

  self.trainData=trainData;
  self.testData=testData;

  -- Creates a confusion matrix object
  self.confusion = optim.ConfusionMatrix(trainData.classNames);
  -- Retrieves parameters and gradients (these are tensors that will be automatically updated during training)
  self.parameters,self.gradParameters = model.net:getParameters();

  -- prints out some information about the network
  local tr_params = "# of trainable parameters: " .. (self.gradParameters:size())[1]
  print("\nCNN using architecture: " ..tostring(model).."")
  print(self.model.net)
  print(tr_params .."\n")

  -- logs utils
  local time = os.date("*t")
  -- id associated to one execution of the script
  local id = time.month .. time.day .. time.hour .. time.min
  local tr_log_f = logDir .. '/' .. tostring(model) ..'_' .. id .. '_train.log'
  local te_log_f = logDir .. '/' .. tostring(model) ..'_' .. id .. '_test.log'
  self.trLog = optim.Logger(tr_log_f)
  self.teLog = optim.Logger(te_log_f)
  self.trLog:setNames{'arch', 'error rate', 'loss', 'train time'}
  self.teLog:setNames{'error rate', 'test time'}
  
  self.totTrainTime=0;

  -- logs out some information about the network
  self.trLog:add{tostring(self.model.net).."\n"..tr_params, '' , '' , '' }

  -- If necessary it optimizes the machine for running on cuda
  if self.inputType == 'cuda' then
    self.model.net:cuda()
    self.model.loss:cuda()
  end

end

--[[ Runs an experiment by performing a set of epochs of training, each followed by a testing ]]
function machine:runExp()
  local maxEpochs=self.maxEpochs
  for epoch=1,maxEpochs do
    self.epoch=epoch

    --    if epoch==10 then
    --      self.batchSize=self.trainData:size()
    --      self.optAlgo=lbfgs()
    --    end
    --    self.batchSize=self.batchSize*epoch

    self:train()
    self:test()
  end
  
  print("Total training time =" .. self.totTrainTime .."s\n\n")
  
end

--[[ Trains the machine for one epoch ]]
function machine:train()

  print("Training using: " .. tostring(self.optAlgo) .. " with learning rate = "..self.optAlgo.learningRate .. "\n")

  local time = sys.clock()
  local ntr=self.trainData:size()
  local loss=0;

  -- sets model to training mode
  self.model.net:training()

  -- shuffles the samples and saves the results
  local shuffle = torch.randperm(ntr)

  -- trains for one epoch
  print(" epoch # " .. self.epoch .. '/' .. self.maxEpochs .. ' [batchSize = ' .. self.batchSize .. ']')
  for t = 1,ntr,self.batchSize do

    -- disp progress
    xlua.progress(torch.ceil(t/self.batchSize), torch.round(ntr/self.batchSize))

    -- creates a mini batch
    local inputs = {}
    local targets = {}

    for i = t,math.min(t+self.batchSize-1,ntr) do

      -- load new sample
      local input = self.trainData.data[shuffle[i]]
      local target = self.trainData.labels[shuffle[i]]

      --converts the sample to the desired format (to get the desired precision)
      if self.inputType == 'double' then input = input:double()
      elseif self.inputType == 'cuda' then input = input:cuda() end

      --adds the selected inputs and targets to the appropriate tables
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    --    p:start('opt','')
    local slck,err = self.optAlgo.optimize(self:getFwdBwdFunc(inputs, targets), self.parameters, self.optAlgo)
    loss= loss + err[1]

  end

  -- computes 1 epoch training time
  time = sys.clock() - time
  self.totTrainTime=self.totTrainTime+time;
    
  -- prints train error rate
  self.confusion:updateValids()
  local err_rate = (1 - self.confusion.totalValid) * 100;
  print('\n train error rate = '..err_rate..'%')

  -- prints time taken
  print(" training time = " .. time .. 's')
  print(" time / sample = " .. (time / ntr*1000) .. 'ms\n')

  -- logs out accuracy, loss and train time
  self.trLog:add{ '', err_rate, loss, time }

  -- next epoch
  self.confusion:zero()
end

--[[ Tests the machine ]]
function machine:test()
  -- local vars
  local time = sys.clock()
  local nte=self.testData:size();

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  self.model.net:evaluate()

  -- test over test data
  print('Testing:')
  for t = 1,nte do
    -- disp progress
    xlua.progress(t, nte)

    -- get new sample
    local input = self.testData.data[t]
    --converts the data sample to the desired format (to get the desired precision)
    if self.inputType == 'double' then input = input:double()
    elseif self.inputType == 'cuda' then input = input:cuda() end
    local target = self.testData.labels[t]

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

--[[ Function used to evaluate the network (forward) on some inputs, compute the loss and the gradient vector (backward) accordingly ]]
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
