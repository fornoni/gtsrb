require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers


--[[ Class used to represent a basic two-stages network]]
local twoStagesNet = torch.class('twoStagesNet')

--[[ Initialization method, assigning the parameters of the two-stages network a default values]]
function twoStagesNet:__init(dataset, optAlgo)
  -- number of classes of the problem
  self.noutputs = #(dataset.classNames);
  -- number of channels of each image
  self.nfeats= (dataset.data:size())[2];
  -- width of each image (height is assumed to be the same)
  self.width=dataset.data:size()[3];

  -- number of kernels / neurons at each layer
  --  self.nmaps = {64,64,128};
  self.nmaps={16,32,64}
  -- size of each kernel
  self.filtsize = {5,5};
  -- pooling stride and size of the kernel
  self.poolsize = {2,2};
  -- default loss function
  self.loss= nn.ClassNLLCriterion();

  -- if true it means that the network has already been turned into a feature extractor
  self.featureExtractor=false
  
  -- specifies the default optimization algorithm for the architecture 
  self.optAlgo=optAlgo or sgd();
  
  -- specifies the default number of layers to be removed to obtain a feature extractor
  self.nFeatsLayers= 3
  
  -- gets the sizese of the convolutional layers (maps and flattened sizes)
  self:getSizes()
end

--[[ calculate the sizes of the convolutional layers and stores them in the appropriate fields ]]--
function twoStagesNet:getSizes()
  -- the size of the feature maps in the first convolutional layer
  self.l0_mapsize=math.floor((self.width-self.filtsize[1]+1)/self.poolsize[1])
  -- the size of the output of the first convolutional layer, when flattened down
  self.l0_flatsize=self.nmaps[1]*self.l0_mapsize*self.l0_mapsize
  -- the size of the feature maps in the second convolutional layer
  self.l1_mapsize=math.floor((self.l0_mapsize-self.filtsize[2]+1)/self.poolsize[2])
  -- the size of the output of the second convolutional layer, when flattened down
  self.l1_flatsize=self.nmaps[2]*self.l1_mapsize*self.l1_mapsize
  
--  print("self.l0_mapsize = " .. self.l0_mapsize)
--  print("self.l0_flatsize = " .. self.l0_flatsize)
--  print("self.l1_mapsize = " .. self.l1_mapsize)
--  print("self.l1_flatsize = " .. self.l1_flatsize)
end

--[[ returns a copy of the network that can be used as a feature extractor. this is achieved by removing the last n_layers
ARGS:
- `n_layers`             : the number of layers to be removed from the network to use it as a feture extractor
]]--
function twoStagesNet:toFeatureExtractor(n_layers)
  n_layers =  n_layers or self.nFeatsLayers

  -- removes last layers
  self:removeLastLayers(n_layers)  
end


--[[removes the last n_layers from the CNN (for feature extraction purposes)
ARGS:
- `n_layers`             : the number of layers to be removed from the network to use it as a feture extractor
]]--
function twoStagesNet:removeLastLayers(n_layers)
  
  print("Removing last " .. n_layers .. " layers")
  
  --removes the last n_layers
  for i=1,n_layers do
    self.net:remove()
  end
end
