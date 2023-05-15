#!/bin/bash

src=./gen/

for file in $src*; do
    echo "Submitting $file"
    sbatch $file
done

