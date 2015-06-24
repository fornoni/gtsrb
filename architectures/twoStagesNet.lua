
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers


--[[ Class used to represent a basic two-stages network]]
local twoStagesNet = torch.class('twoStagesNet')
--twoStagesNet = {}
--twoStagesNet.__index = twoStagesNet
--setmetatable(twoStagesNet, {
--  __call = function (cls, ...)
--    local self = setmetatable({}, cls)
--    self:__init(...)
--    return self
--  end,
--})

--[[ Initialization method, assigning the parameters of the two-stages network a default values]]
function twoStagesNet:__init(dataset)
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
  self.poolsize = 2;
  -- default loss function
  self.loss= nn.ClassNLLCriterion();

  -- the size of the feature maps in the first convolutional layer
  self.l0_mapsize=((self.width-self.filtsize[1]+1)/self.poolsize)
  -- the size of the feature maps in the second convolutional layer 
  self.l1_mapsize=((self.l0_mapsize-self.filtsize[2]+1)/self.poolsize)
  -- the size of the output of the second convolutional layer, when flattened down 
  self.l1_flatsize=self.nmaps[2]*self.l1_mapsize*self.l1_mapsize
  
  -- if true it means tha the network has already been turned into a feature extractor
  self.featureExtractor=false
end

--[[ returns a copy of the network that can be used as a feature extractor. this is achieved by removing the last n_layers 
ARGS:
- `n_layers`             : the number of layers to be removed from the network to use it as a feture extractor
]]--
function twoStagesNet:toFeatureExtractor(n_layers)
 --clones the network and removes last layers
  self:removeLastLayers(n_layers)
end
