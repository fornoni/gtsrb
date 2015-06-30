require 'optim'
require 'torch'
require 'nn'

function min(x,y)
  return (x < y) and x or y
end

--[[ my Lua/Torch7 implementation of exact kNN. For efficiency, the kNN evaluation is performed in batches.
ARGS:
- `k`                     : the number of nearest neighbors to be used
- `trainData`             : training dataset object
- `testData`              : testing dataset object
]]
function kNN(k,trainData,testData, batchSize)

  -- Creates a confusion matrix object
  local confusion = optim.ConfusionMatrix(trainData.classNames);

  local nte =testData:size()
  local ntr =trainData:size()
  local d = testData.data:size()[2]
  local ncla=#(trainData.classNames)

  -- transposes the training data once, for efficiency
  local tr_data = trainData.data:transpose(1,2)
  -- computes the norm of training features
  local tr_nrm =torch.sum(torch.pow(trainData.data,2),2):transpose(1,2)

  --  local true_slice_size, D, dotprod, testSlice, te_slice_nrm
  --  local D=torch.Tensor(batchSize,ntr)
  --  local dotprod=torch.Tensor(batchSize,ntr)
  --  local testSlice = torch.Tensor(batchSize,d)
  --  local te_slice_nrm=torch.Tensor(batchSize)

  print('exact ' .. k .. '-NN classification in batches of ' .. batchSize .. ' samples')

  for i=1,nte,batchSize do
    --[[ calling the garbage collector at the beginning of each epoch seems to avoid huge memory leak problems.
    I don't really understand why this is necesary, as the variables are declared as "local", in a specific block of code
    and they go out out of scope at the end of each loop.
    ]] 
    collectgarbage()

    xlua.progress(torch.ceil(i/batchSize), torch.round(nte/batchSize))

    -- first test using iterative method: too slow
    --    print("doing " .. torch.ceil(i/batchSize) .. "/" .. torch.round(nte/batchSize))
    --    local d=torch.Tensor(ntr):zero()
    --    for j=1,ntr do
    --      d[j]=torch.sum(torch.pow((testData.data[i] - trainData.data[j]),2))
    --    end

    do
      --gets the current slice of the testing data
      local true_slice_size=min(nte,i+batchSize-1)-i+1;
      local testSlice=testData.data[{{i,min(nte,i+batchSize-1)},{}}]

      -- computes Euclidean distance of the testing slice to the training data in efficient matrix form
      -- first computes the norm of the testing samples
      local te_slice_nrm = torch.sum(torch.pow(testSlice,2),2)
      -- then computes the dot product between the testing slice samples and the training samples
      local dotprod = torch.mm(testSlice, tr_data)
      -- then combines the different quantities to get D
      local D = tr_nrm:expand(true_slice_size, ntr) + te_slice_nrm:expand(true_slice_size, ntr) - torch.mul(dotprod, 2)

      local ind;
      
      -- computes the closest index(es)
      if k==1 then
        _,ind = torch.min(D, 2)
      else
        _,ind= torch.sort(D,2)
      end

      for j = 1,true_slice_size do
        -- for each sample performs the kNN prediction
        local ypred;
        if k==1 then
          ypred = trainData.labels[ind[j][1]]
        else
          --vector of votes
          local votes= torch.Tensor(ncla):zero()
          for h=1,k do
            local k_samp=ind[j][h]
            local k_class=trainData.labels[k_samp]
            votes[k_class] = votes[k_class] +1
          end
          _,ypred = torch.max(votes, 1)
          ypred=ypred[1]
        end
        --      xlua.progress(i+j-1,nte)
        confusion:add(ypred, testData.labels[i+j-1])
      end
    end

  end

  confusion:updateValids()
  local err_rate = (1 - confusion.totalValid) * 100;

  print("\n error rate: " .. err_rate .. "\n")
  return err_rate
end


--[[ Class representing a sgd algorithm]]
local sgd = torch.class('sgd')
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
--[[ Returns a string representing the hyperparameters of the object ]]--
function sgd:parstostring()
  return "LR"..self.learningRate .. "_LRD" .. self.learningRateDecay .. "_MM"..self.momentum .."_WD"..self.weightDecay
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

--[[ Returns a string representing the hyperparameters of the object ]]--
function lbfgs:parstostring()
  return "LR"..self.learningRate .. "_MI" .. self.maxIter .. "_NC"..self.nCorrection
end