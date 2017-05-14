#!/bin/bash

# Clears local processes on ports in case of error

for x in `seq 6000 6010`
do
    echo $x
    echo `lsof -t -i:$x`
done

for x in `seq 6000 6010`
do
    kill `lsof -t -i:$x`
done
