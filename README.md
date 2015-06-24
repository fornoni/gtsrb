# GTSRB Challenge

a.k.a German Traffic Sign Recognition Benchmark :de: :no_entry: :no_bicycles:
:no_entry_sign: ...

## Usage
Main pre-requisite: torch7. Additional pre-requisites for the download script: bash, wget, unzip. 

1. Clone the repository
2. Download and unpack the data, using:
	`$ getData.sh some_dir`
   where some_dir is a directory where you want the dataset to be downloaded / unzipped
3. Run the Convolutional Neural Network, using:
	`$ th main.lua -data some_dir`

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
