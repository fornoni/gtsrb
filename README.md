# GTSRB Challenge

a.k.a German Traffic Sign Recognition Benchmark :de: :no_entry: :no_bicycles:
:no_entry_sign: ...

## Usage
Main pre-requisite: torch7. Additional pre-requisites for the download script: bash, wget, unzip. 

1. Clone the repository
2. Download and unpack the data, using:
	`$ getData.sh some_dir`
   where some_dir is a directory in which you want the dataset to be downloaded / unzipped
3. Run the Convolutional Neural Network (CNN), using:
	`$ th runCNN.lua -data some_dir -epochs n_epochs -arch mpNet`
   where n_epochs is the total number of training epochs desired and arch is the type of architecture desired (for the moment only mpNet, or tinyNet).
   It is possible to resume the training of a partially trained network by adding the option '-load'. In this case the framework will load the last 
   saved network of desired type and continue the training until the specified number of training epochs is reached.
4. After a CNN has been trained it is also possible to use the last learned model as a feature extractor, together with a simple NN classifier, to produce the final classification.
   The feature extraction + NN classification can be achieved using:
   `$ th runNN.lua -data some_dir -epochs n_epochs -arch mpNet -nl n_layers`   
   where n_layers specifies the number of *non-convolutional* layers to be removed from the CNN to produce a feature extractor, while the other parameters have the same meaning as before
   It is also possible to evaluate the performance the pre-processed image features when directly fed as input to the NN classifier. This can be achieved using the '-orig' option.
   
## Goal

Use [Torch](http://torch.ch/) to train and evaluate a 2-stage convolutional
neural network able to classify German traffic sign images (43 classes):

* fork the repository under your account,
* go to Settings > Features and enable Issues,
* create an issue under your repo describing your approach,
* report your result(s),
* commit your code,
* edit the README with pre-requisites and usage,
* boost accuracy by experimenting the multi-scale architecture,
* compare with the results obtained in matching mode (i.e use the features with a distance-based search).

## Paper

[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/reference_papers/2-sermanet-ijcnn-11-mscnn.pdf), by Yann LeCun et al.

## Dataset

### Training

`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip` (263 MB)

### Testing

`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip` (84 MB)
`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip` (98 kB)
