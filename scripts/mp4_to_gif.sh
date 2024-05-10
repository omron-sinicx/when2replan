#!/bin/bash

# Convert all mp4 files in the folder to gif files by ffmpeg

# Get arguments
FOLDER=$1

# get files in the folder
FILES=$FOLDER/*.mp4

# convert all mp4 files to gif files
for f in $FILES
# filename without extension
do
  filename=$(basename -- "$f")
  # take action on each file. $f store current file name
  echo "Processing $filename"
  # saved file path
  saved_path=$FOLDER/${filename%.*}.gif
  # take action on each file. $f store current file name
  ffmpeg -i $f -y $saved_path
done
