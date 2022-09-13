#!/bin/bash

cd $(dirname "$BASH_SOURCE")

sudo apt update
sudo apt install nodejs npm

npm i

if [ ! -d "weights" ]; then
  mkdir weights
fi
cd weights
wget https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json
wget https://github.com/justadudewhohacks/face-api.js/raw/master/weights/tiny_face_detector_model-shard1
wget https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_landmark_68_model-shard1
wget https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-weights_manifest.json
wget https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_recognition_model-weights_manifest.json
wget https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_recognition_model-shard2
wget https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_recognition_model-shard1
wget https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_expression_model-shard1
wget https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-weights_manifest.json
