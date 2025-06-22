#!/bin/bash

echo "this is b.sh"
workDir=`pwd`
# echo ${workDir}
timestamp=`date +%Y%m%d%H%M%S`
sh ${workDir}/scripts/test/a.sh ${timestamp} ${workDir}
# sh a.sh
