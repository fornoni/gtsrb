require "architectures.twoStagesNet"

--[[ Class used to represent a MaxPooling two-stages network. It inherits from twoStagesNet]]
local mpNet = torch.class('mpNet', 'twoStagesNet')

--[[ Initialization method, assigning the parameters of the two-stages network a default values]]
function mpNet:__init(dataset)
  twoStagesNet:__init(dataset) -- calls the base class constructor
  self.connTableDensity=1;
  self.dropoutP={.4,.4,.4};
  
  -- specifies the default optimization algorithm for the architecture 
  self.optAlgo=sgd();
  self.optAlgo.learningRate=5e-3
  self.optAlgo.learningRateDecay=1e-5
end

--[[ Function that returns a torch7 network for the specified architecture ]]--
function mpNet:getModel()
  self:getSizes()
  
  self.net = nn.Sequential()

  -- first convolutional stage, with dropout, rectification non-linearity and max pooling
  self.net:add(nn.SpatialConvolutionMM(self.nfeats, self.nmaps[1], self.filtsize[1], self.filtsize[1]))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxPooling(self.poolsize[1],self.poolsize[1],self.poolsize[1],self.poolsize[1]))
  self.net:add(nn.Dropout(self.dropoutP[1]))

  -- second convolutional stage, with dropout, rectification non-linearity and max pooling
  -- if densely connected it uses SpatialConvolutionMM, otherwise it uses a random connection table
  if(self.connTableDensity==1) then
    self.net:add(nn.SpatialConvolutionMM(self.nmaps[1], self.nmaps[2], self.filtsize[2], self.filtsize[2]))
  else
    local table = nn.tables.random(self.nmaps[1],self.nmaps[2],torch.round(self.nmaps[1]*self.connTableDensity))
    self.net:add(nn.SpatialConvolutionMap(table, self.filtsize[2], self.filtsize[2]))
  end
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxPooling(self.poolsize[2],self.poolsize[2],self.poolsize[2],self.poolsize[2]))
  self.net:add(nn.Dropout(self.dropoutP[2]))

  -- final densely connected layers, again with dropout and rectification nonlinearity
  self.net:add(nn.View(self.nmaps[2]*self.l1_mapsize*self.l1_mapsize))  
  self.net:add(nn.Linear(self.nmaps[2]*self.l1_mapsize*self.l1_mapsize, self.nmaps[3]))
  self.net:add(nn.ReLU())
  self.net:add(nn.Dropout(self.dropoutP[3]))
  self.net:add(nn.Linear(self.nmaps[3], self.noutputs))
  self.net:add(nn.LogSoftMax())

  -- negative log-likelihood loss function
  self.loss = nn.ClassNLLCriterion()

  -- Print the size of the Threshold outputs
  --  local conv_nodes = self.net:findModules('nn.SpatialConvolution')
  --  for i = 1, #conv_nodes do
  --    print("layer "..i.." size: ")
  --    print(conv_nodes[i].output:size())
  --  end

end