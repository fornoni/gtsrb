#!/bin/bash
if [ $# -eq 0 ];  then 
	echo "Error! you should provide a string commit message";
	exit 1;
fi

git add *.lua
git add *.md
git add *.sh
git add architectures/mpNet.lua
git add architectures/twoStagesNet.lua

git commit -m $1
git push
