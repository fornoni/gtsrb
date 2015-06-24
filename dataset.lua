require 'torch'
require 'nn'
require 'image'
require 'csv'

--[[ Class used to represent a dataset]]
local dataset = torch.class("dataset")
--dataset = {}
--dataset.__index = dataset
--setmetatable(dataset, {
--  __call = function (cls, ...)
--    local self = setmetatable({}, cls)
--    self:__init(...)
--    return self
--  end,
--})

--[[ Initializes a new instance of a dataset]]
function dataset:__init(obj)
  self.data = obj and obj.data or torch.Tensor(0);
  self.labels = obj and obj.labels or torch.Tensor(0);
  self.classNames = obj and obj.classNames or {};
  self.fileNames= obj and obj.fileNames or {};
  self.imgExt=obj and obj.imgExt or 'ppm'
end

--[[ Returns the size of a dataset, i.e. the number of images it contains]]
function dataset:size()
  return (self.data:size())[1]
end

--[[ Returns the dimensionality of a dataset, i.e. the resolution of the images it contains]]
function dataset:dim()
  return (self.data:size())[{2,-1}]
end

--[[ Bulk loads the images into dataset, rescaling all the images to 32x32 converting them to YUV and retaining only the Y component
ARGS:
- `imgs_dir`              : directory containing the images
- `images_list_file`      : (optional) if sepcified it loads only the images specified in the csv file
]]
function dataset:loadImgs(imgs_dir,images_list_file)
  local labels={};

  if images_list_file then
    --it reads the images from the specified csv file  
    local separator = ';' -- optional; use if not a comma
    local csv = Csv(images_list_file, "r",separator)
    local header = csv:read() -- header is an array of strings

    local tmp={};
    while true do
      local tmp = csv:read()
      if not tmp then break end
      local file_path=paths.concat(imgs_dir,tmp[1]);
      table.insert(self.fileNames, file_path)
      local lbl=tonumber(tmp[8]) +1;
      table.insert(labels, lbl)
      if not self.classNames[lbl] then
        self.classNames[lbl]=lbl;
      end
    end
    csv:close()
  else
    --it scrolls the subdirectories of the images directory and stores the file paths in a table
    for class_dir in paths.files(imgs_dir) do
      if class_dir~='.' and class_dir~='..' then

        print("processing folder: "..class_dir)
        local lbl=tonumber(class_dir)+1
        if not self.classNames[lbl] then
          self.classNames[lbl]=lbl;
        end
        --it goes over all files in the directory
        for file in paths.files(imgs_dir .. "/" .. class_dir) do
          --        print( file )
          --it loads all the files matching the extension
          if file:find(self.imgExt .. '$') then
            local file_path=paths.concat(paths.concat(imgs_dir,class_dir),file);
            table.insert(self.fileNames, file_path)
            table.insert(labels, lbl)
          end
        end
      end
    end
  end

  --checks if files were found
  if #self.fileNames == 0 then
    error('The given directory doesn\'t contain any files of type: ' .. self.imgExt)
  end
  
  --shapes the tensors containing the images and the labels
  self.data=torch.Tensor(#self.fileNames,1,32,32)
  self.labels=torch.Tensor(#self.fileNames)

  print 'Loading images..'

  --it goes over the file tables and loads the corresponding images into the self.data field:
  for i,file in ipairs(self.fileNames) do
    -- loads each image
    print( 'loading ' .. i ..'-th img: ' .. file )
    self.data[{i,1,{},{}}]=image.rgb2y(image.scale(image.load(file),32,32))
    self.labels[i]=labels[i]
  end
end


--[[ Post-processes the images by:
      1) converting them to YUV and retaining only the Y component
      2) performing global whitening (zero mean and unit variance) on the pixel distribution
      3) performing local spatial contrastive normalization 
ARGS:
- `mean`     : the mean to be used to whiten the pixel distribution; if not specified it is computed from the data
- `std`      : the mean to be used to whiten the pixel distribution; if not specified it is computed from the data
]]
function dataset:postProc(mean,std)

  --converts the data to float
  self.data = self.data:float()

  local mean = mean or self.data[{ {},1,{},{} }]:mean()
  local std = std or self.data[{ {},1,{},{} }]:std()

  --global whitening
  print 'preprocessing: making the Y values zero-meaned and unit varianced'
  print(mean)
  print(std)
  self.data[{ {},1,{},{} }]:add(-mean)
  self.data[{ {},1,{},{} }]:div(std)

  --local normalization
  print 'preprocessing: using spatial contrastive normalization on the Y channel'
  -- Define the normalization neighborhood:
  local neighborhood = image.gaussian1D(13)
  local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
  for i = 1,self:size() do
    self.data[{ i,1,{},{} }] = normalization:forward(self.data[{ i,1,{},{} }]:resize(1,32,32))
  end

  return mean, std
end
