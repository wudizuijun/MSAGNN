#!/bin/bash
echo "this is a.sh"
echo "get args $1"
runingTime=$1
workDir=$2

python ${workDir}/scripts/test/test_args.py --time ${runingTime}
    