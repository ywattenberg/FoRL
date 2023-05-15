#!/bin/bash

src=$HOME/forl/FoRL/gen/

bash ./euler_generate_jobs.sh

for file in $src*; do
    echo "Submitting $file"
    bash $file
    exit 0
done

