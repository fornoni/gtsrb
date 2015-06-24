
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

  -- the size of the first convolutional layer
  self.l0_mapsize=((self.width-self.filtsize[1]+1)/self.poolsize)
  -- the size of the second convolutional layer 
  self.l1_mapsize=((self.l0_mapsize-self.filtsize[2]+1)/self.poolsize)
end
