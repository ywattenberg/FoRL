#!/bin/bash

src=$HOME/forl/FoRL/gen/

bash ./euler_generate_jobs.sh

for file in $src*; do
    echo "Submitting $file"
    sbatch $file
done

