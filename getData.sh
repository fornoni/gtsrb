#!/bin/bash

if [ $# -ge 1 ]; 
    then
	data_dir=$1;
    else
	echo "error: you must specify a directory where to download the data";
	exit;
fi


if [ ! -d "${data_dir}" ]; then
  if mkdir -p ${data_dir} ; then
	echo "created directory: " ${data_dir}
  else
    echo "error: impossible to create directory";
    exit
  fi
fi

cwd=`pwd`
cd ${data_dir}

if [ ! -d train ]; then
	mkdir train
fi
if [ ! -d test ]; then
	mkdir test
fi

wget -c http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
wget -c http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
wget -c http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip


unzip GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_GT.zip -d ./test/

mv GTSRB/Final_Training ./train/.

mv GTSRB/Final_Test ./test/.

rm -R GTSRB

cd $cwd
