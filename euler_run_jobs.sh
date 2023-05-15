#!/bin/bash

src=$HOME/forl/FoRL/gen/

for file in $src*; do
    echo "Submitting $file"
    sbatch $file
    exit 0
done

