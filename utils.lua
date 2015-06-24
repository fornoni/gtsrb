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
