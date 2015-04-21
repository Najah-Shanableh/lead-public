#!/bin/bash

# is specified path is a directory or does not exist
#if [ -d "$2" ] || [ ! -e "$2" ]
#then
#	basedir=$2 #/$(date "+%Y%m%d%H%M%S")
#	mkdir -p $basedir
	
#	echo Writing csv to $basedir ...
#	./write_data.py $basedir $1 || { exit 1; }
#else
#	basedir=$(dirname $2)
#fi

mkdir -p $3
cp $1 $3/params.yaml # move the params file to the subdir

./n_models.py $1

rm -rf $3/*/

./get_params.py $1 $2 $3 | parallel --delay 5 --joblog $3/log ./run_model.py
