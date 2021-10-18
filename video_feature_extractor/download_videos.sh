#!/bin/bash
while read line; do gsutil cp -r $line ./input; done < videos.txt