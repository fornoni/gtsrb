require 'torch'


--[[ Utility function to load the training and testing images, preprocess them and serialize the results to a file ]]
function geGTSRtData(data_dir)

  -- directories where the training/testing images and metadata lie
  local tr_im_dir=data_dir .. '/train/Final_Training/Images'
  local te_im_dir=data_dir .. '/test/Final_Test/Images'
  local te_csv_file=data_dir..'/test/GT-final_test.csv'

  --files where to save the post-processed images
  local tr_dat=data_dir .. '/train.dat'
  local te_dat=data_dir .. '/test.dat'

  local trainData, testData, mean, std

  --if the training and testing files do not already exist it loads the images, pre-process them and save the results
  if not file_exists(tr_dat) or not file_exists(te_dat) then
    trainData=dataset()
    trainData:loadImgs(tr_im_dir)
    print('Loaded training images')

    testData=dataset()
    testData:loadImgs(te_im_dir, te_csv_file)

    mean,std=trainData:postProc()
    testData:postProc(mean,std)

    print('trainData:size()=' .. trainData:size())

    torch.save(tr_dat, trainData)
    torch.save(te_dat, testData)
  else
    --otherwise it simply loads the pre-processed data
    trainData=dataset(torch.load(tr_dat))
    testData=dataset(torch.load(te_dat))
  end

  return trainData,testData
end

--[[ Utility function that creates training/testing feature dirs and returns their paths]]
function getFeatDirs(data_dir)
  local tr_feat_dir = data_dir .. '/train/Final_Training' .. '/Features'
  local te_feat_dir = data_dir .. '/test/Final_Test/' .. '/Features'
  --if it doesn't exists it creates the train feature dir
  if not file_exists(tr_feat_dir) then
    os.execute("mkdir " .. tr_feat_dir)
  end
  --if it doesn't exists it creates the test feature dir
  if not file_exists(te_feat_dir) then
    os.execute("mkdir " .. te_feat_dir)
  end

  return tr_feat_dir,te_feat_dir
end

--[[ Utility function to perform feature extraction using a given CNN and save the resulting datasets 
ARGS:
- `convnet`              : a machine object (CNN) to be used as a feature extractor  
- `n_layers`             : the number of layers to be removed from the network to use it as a feture extractor
- `trainData`            : training dataset object
- `testData`             : testing dataset object
- `data_dir`             : the main data directory of GTSR
]]
function extracFeats(convnet, n_layers, trainData, testData, data_dir)
  local tr_feat_dir, te_feat_dir = getFeatDirs(data_dir)
  local te_feat_file = te_feat_dir .. "/"  .. convnet.machineId .. "_L" .. n_layers .. ".feat"
  local tr_feat_file = tr_feat_dir .. "/"  .. convnet.machineId .. "_L" .. n_layers .. ".feat"
  --if the training and testing files do not already exist it extracts the features
  if not file_exists(tr_feat_file) or not file_exists(te_feat_file) then
    --extract and saves test features
    testData.data = convnet:extractFeats(n_layers,testData)
    print("Saving testing features from "..te_feat_file)
    torch.save(te_feat_file, testData)

    --extract and saves train features
    trainData.data = convnet:extractFeats(n_layers,trainData)
    print("Saving training features from "..tr_feat_file)
    torch.save(tr_feat_file, trainData)
  else
    --otherwise it simply loads the pre-processed data
    print("Loading testing features from "..te_feat_file)
    testData=dataset(torch.load(te_feat_file))
    print("Loading training features from "..tr_feat_file)
    trainData=dataset(torch.load(tr_feat_file))
  end
  
  return trainData, testData
end


--[[ Checks if a file exists ]]
function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then
    io.close(f)
    return true
  else
    return false
  end
end
