#!/bin/bash

IFS="/"
for filename in Augmented/*; do
    read -a array <<< "$filename"
    echo ${array[1]}
    python resnet-50-2.py ${array[1]} 100 
done
