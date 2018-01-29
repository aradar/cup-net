# Introduction
Our goal with this project was to build our own dataset to train a neural network to detect blue and yellow cups which 
are upside down. The resulting network was supposed to run on a raspberry pi 3 and help our autonomous car detect its 
track (blue and yellow cups).

We took around 3500 images and labeled 1300 of those. After that we decided to use a mobilenet v1 which was already 
trained based on the COCO dataset. We then started to train the network using our own dataset.

# Train the network
Following steps are needed to continue the training of the pre trained network.
 
- complete the setup part first if you want to do this!
- clone this repo
- go into the `models/mobilenet_ssd` dir
- download the pre trained 
  [mobilenet-v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz)
- extract the downloaded network into a dir called `ssd_mobilenet_v1_coco_2017_11_17` in the `models/mobilenet_ssd` dir 
- start the training with `python3 $TF-MODELS/research/object_detection/train.py --pipeline_config_path=pipeline_config.pbtext --train_dir=./train/`
  - `$TF-MDOELS` should point to the location of the tensorflow models repo (it does not have to be a env variable). 

# Build the dataset
This section guides you on how to build the dataset based on the images and labels in the braindead_v.2.zip.  

- complete the setup part first if you want to do this!
- clone this repo
- download [braindead_v.2.zip](https://drive.google.com/open?id=1DrooMH7rW0lPt1ksN-CUVWA8axUqVSwO) and put it in the 
  root dir of the repo
- cd into the root of the repo
- execute the following commands
  - `mdkir -p raw_data/source`
  - `unzip braindead_v.2.zip -d raw_data/source` 
  - `build_image_structure.py ./raw_data/source ./raw_data -c 1000 --strict`
    - puts random images from `./raw_data/source` into folders for training and evaluation. The default split results 
      in 90% training and 10% evaluation (this and a lot more can be changed through parameters see --help for this).
  - `pascal_voc_xml_to_tfrecord.py --image_path ./raw_data/train/image --annotations_path ./raw_data/train/annotation --output_file ./data/train.tfrecord`
    - builds train.tfrecord based on the images and labels in `./raw_data/train`
  - `pascal_voc_xml_to_tfrecord.py --image_path ./raw_data/eval/image --annotations_path ./raw_data/eval/annotation --output_file ./data/eval.tfrecord`
    - builds eval.tfrecord based on the images and labels in `./raw_data/eval`

# Setup
Steps which are needed to work with this repo.

- clone https://github.com/tensorflow/models

## Getting the tensorflow models repo up and running
- Clone https://github.com/tensorflow/models
- cd into the cloned dir and there into the research folder
- install the models with `pip install .`

## Things which have to be done before you can use cup-net \(WIP\)
- install slim in research
- compile the protobuff files
- install tk package
- download the pretrained model

# Available resources
## Raw dataset (images and labels)
- Images with labels ([braindead.zip](https://drive.google.com/open?id=17HI1BiTli7hxVPEBeFtmVkElx69I3wcB), 
    [braindead_v.2.zip](https://drive.google.com/open?id=1DrooMH7rW0lPt1ksN-CUVWA8axUqVSwO))
- Only images ([picam.zip](https://drive.google.com/open?id=1NVJyT4RVjYXzbcBS8EJ_r9ke8ykJtfLP))

## Prebuilt dataset (train.tfrecord + eval.tfrecord)
train.tfrecord contains 900 images each with a label and eval.tfrecord contains 100 images also with labels.

- [train.tfrecord](https://drive.google.com/open?id=1vRQSt3LNW-b2WB2mt7zziW9fa5Ecdxze)
- [eval.tfrecord](https://drive.google.com/open?id=1VDs20g2l5ohqR78Nk5DGEeGI3x_IGUh5)

## Already trained network (3000 epochs)
The following link gets you an already trained mobilenet-v1 with our dataset.
\
\
[cup-net.zip](https://drive.google.com/open?id=19z2xIgi6mrl1X8Y8D8FAoLoI5aE-IZt5)