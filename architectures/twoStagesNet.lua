
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
  self.nfeats= (dataset.data:size())[2];
  self.width=dataset.data:size()[3];

--  self.nmaps = {64,64,128};  
  self.nmaps={16,32,64}
  self.filtsize = {5,5};
  self.poolsize = 2;
  self.loss= nn.ClassNLLCriterion();
  
  self.l0_mapsize=((self.width-self.filtsize[1]+1)/self.poolsize)
  self.l1_mapsize=((self.l0_mapsize-self.filtsize[2]+1)/self.poolsize)
end