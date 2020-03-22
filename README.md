# Perception

This project is intended to implement some of environment 
perception tasks that are often used in Autonomous Vehicles pipelines.

Project consists of 3 parts:
* Detecting objects with neural network
* Estimating image depth from corresponding pairs of stereo images
* Computing distances to closest obstacles using image depths and bounding boxes

Experiments are based on famous road traffic Kitti 2D object detection dataset
that consists of 7481 training images and 7518 test images:
http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

Example of image:
![Test Image 1](https://github.com/ieBoytsov/perception/images/sample_img.png)

Given images from both left and right camera it is possible...


