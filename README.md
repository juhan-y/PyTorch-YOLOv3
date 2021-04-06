## kookmin university '차량지능기초' first assignment.
## Hi I'm juhan yoon. there's summary below the lines. please read the lines before see my sources.

# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

## There's a lot of codes in this original github, so i couldn't write explanation of all files(python file, config, etc..).

## So you just watch codes in train, test, models python file. there are explanation I wrote myself.

## Explanation of this project
I used PyTorch-YOLOv3 to implement computer vision for self-driving in deep-learning.( using google colaboratory)
The original doucumnet recommended using COCO dataset, but I don't want to use typical dataset.
so I used vehicle-openimages dataset in roboflow. you can find out what it is in https://public.roboflow.com/object-detection/vehicles-openimages
This dataset doesn't have many images, there're 627 images including test, train, dev set.
So I trained my model more than usual. please keep in your mind and see my model.

## Address of colab and original document I forked from.
https://colab.research.google.com/drive/1tFRKda4d6vKD6geag0WNzkU8y5DA8X9U?usp=sharing For creating model, I wrote codes on Colab.
https://github.com/eriklindernoren/PyTorch-YOLOv3 erik lindernoren ML engineer at Apple.
