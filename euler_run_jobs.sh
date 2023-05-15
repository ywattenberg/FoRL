#!/bin/bash

src=$HOME/forl/FoRL/gen/

for file in $src*; do
    echo "Submitting $file"
    bash $file
    exit 0
done

