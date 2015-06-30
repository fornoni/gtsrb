require "architectures.twoStagesNet"

--[[ Class used to represent a MultiScale two-stages network. It inherits from twoStagesNet]]
local msNet = torch.class('msNet', 'twoStagesNet')

--[[ Initialization method, assigning the parameters of the two-stages network a default values]]
function msNet:__init(dataset)
  twoStagesNet:__init(dataset) -- calls the base class constructor
  self.connTableDensity=1;
  self.dropoutP={.2,.4,.4};
  
  -- specifies the default optimization algorithm for the architecture 
  self.optAlgo=sgd();
  self.optAlgo.learningRate=6e-3
  self.optAlgo.learningRateDecay=1e-5
end

--[[ Function that returns a torch7 network for the specified architecture ]]--
function msNet:getModel()
  self:getSizes()

  -- creates a new sequential NN architecture
  self.net = nn.Sequential()

  -- it drops the input with a small probability
  --  self.net:add(nn.Dropout(self.dropoutP[1]))

  -- adds a first convolutional stage, with rectification non-linearity and max pooling
  self.net:add(nn.SpatialConvolutionMM(self.nfeats, self.nmaps[1], self.filtsize[1], self.filtsize[1]))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxPooling(self.poolsize[1],self.poolsize[1],self.poolsize[1],self.poolsize[1]))

  -- creates the second convolutional stage, with rectification non-linearity and max pooling
  local snd = nn.Sequential()
  -- if densely connected it uses SpatialConvolutionMM, otherwise it uses a random connection table
  if(self.connTableDensity==1) then
    snd:add(nn.SpatialConvolutionMM(self.nmaps[1], self.nmaps[2], self.filtsize[2], self.filtsize[2]))
  else
    local table = nn.tables.random(self.nmaps[1],self.nmaps[2],torch.round(self.nmaps[1]*self.connTableDensity))
    snd:add(nn.SpatialConvolutionMap(table, self.filtsize[2], self.filtsize[2]))
  end
  snd:add(nn.ReLU())
  snd:add(nn.SpatialMaxPooling(self.poolsize[2],self.poolsize[2],self.poolsize[2],self.poolsize[2]))
  snd:add(nn.View(self.l1_flatsize))

  --    local frst = nn.Sequential()
  --    -- second convolutional stage, with dropout, rectification non-linearity and max pooling
  --    -- if densely connected it uses SpatialConvolutionMM, otherwise it uses a random connection table
  --    if(self.connTableDensity==1) then
  --      frst:add(nn.SpatialConvolutionMM(self.nmaps[1], self.nmaps[2], self.filtsize[2], self.filtsize[2]))
  --    else
  --      local table = nn.tables.random(self.nmaps[1],self.nmaps[2],torch.round(self.nmaps[1]*self.connTableDensity))
  --      frst:add(nn.SpatialConvolutionMap(table, self.filtsize[2], self.filtsize[2]))
  --    end
  --    frst:add(nn.ReLU())
  --    frst:add(nn.SpatialMaxPooling(self.poolsize[2],self.poolsize[2],self.poolsize[2],self.poolsize[2]))
  --    frst:add(nn.Dropout(self.dropoutP[2]))

  -- creates the (flattened) view of the output of the first convolutional stage that will be fed to the classifier
  local frst = nn.View(self.l0_flatsize)

  -- creates a concatenation of the output of the first convolutional stages and the second convolutional stage
  local cc = nn.ConcatTable()
  cc:add(frst)
  cc:add(snd)

  -- adds the concatenated outputs to the network, and a dropout regularization "layer"
  self.net:add(cc)
  self.net:add(nn.JoinTable(1))
  self.net:add(nn.Dropout(self.dropoutP[2]))

  -- final densely connected layers, again with dropout and rectification nonlinearity
  self.net:add(nn.Linear(self.l0_flatsize + self.l1_flatsize, self.nmaps[3]))
  self.net:add(nn.ReLU())
  self.net:add(nn.Dropout(self.dropoutP[3]))
  self.net:add(nn.Linear(self.nmaps[3], self.noutputs))
  self.net:add(nn.LogSoftMax())

  -- Print the size of the Threshold outputs
  --  local conv_nodes = self.net:findModules('nn.SpatialConvolution')
  --  for i = 1, #conv_nodes do
  --    print("layer "..i.." size: ")
  --    print(conv_nodes[i].output:size())
  --  end

end
