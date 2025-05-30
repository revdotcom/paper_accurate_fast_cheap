#!/bin/bash

ifn=$1
ofn=$2

while read -r line; do
v=$(echo $line | cut -d " " -f 1); 
fn=$(echo $line | cut -d " " -f 2); 
d=$(soxi -D $fn); 
echo $v $d; done < $ifn > $ofn
