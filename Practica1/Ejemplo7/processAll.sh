#!/bin/bash

regex="\.\/images\/img([0-9]*)\.png"
[[ -d out ]] || mkdir -p out
for i in ./images/img*.png; do
    if [[ $i =~ $regex ]]; then
        echo "Processing file $i"
        ./image $i c "./out/cpu${BASH_REMATCH[1]}.png"
        ./image $i g "./out/gpu${BASH_REMATCH[1]}.png"
    else
        echo "$i bad regex" >&2
    fi

    echo
done
