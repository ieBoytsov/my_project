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
![ sample_img](https://user-images.githubusercontent.com/61888740/77248259-2c4ede00-6c49-11ea-9eda-b28a1a932ad1.png)

Given images from both left and right camera it is possible to extract depth

![sample_depth](https://user-images.githubusercontent.com/61888740/77248337-bbf48c80-6c49-11ea-9102-d3947d1765a3.png)

