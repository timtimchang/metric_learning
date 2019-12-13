#!bin/bash

for file in ./*_expectation.py
do
	#file= ${file%.*}
	for i in {0..9}
	do
		echo "$file $i"
		python3 $file $i 
	done

done
