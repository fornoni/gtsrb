require "architectures.mpNet"

--[[ Class used to represent a max-pooling two-stages network. It inherits from twoStagesNet]]
local tinyNet = torch.class('tinyNet', 'mpNet')
--mpNet = {}
--mpNet.__index = mpNet
--setmetatable(mpNet, {
--  __index = twoStagesNet, -- this is what makes the inheritance work
--  __call = function (cls, ...)
--    local self = setmetatable({}, cls)
--    self:__init(...)
--    return self
--  end,
--})

--[[ Initialization method, assigning the parameters of the two-stages network a default values]]
function tinyNet:__init(dataset)
  mpNet:__init(dataset) -- calls the base class constructor
  self.nmaps={4,8,16}
end

