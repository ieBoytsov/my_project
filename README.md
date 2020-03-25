# Perception

This project is intended to implement some of environment 
perception tasks that are often used in Autonomous Vehicles pipelines.

Project consists of the following tasks:
* Detecting objects with neural network (work in progress)
* Estimating image depth from corresponding pairs of stereo images
* Computing distances to closest obstacles using image depths and bounding boxes
* Estimating vehicle trajectory from subsequent images

Some of the tasks are independent while others use output of another tasks.

Experiments are based on famous road traffic Kitti 2D object detection dataset
that consists of 7481 training images and 7518 test images:
http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

Example of image:
![new_img](https://user-images.githubusercontent.com/61888740/77573415-b3b67e80-6ee1-11ea-9d6a-01c07f203211.png)

Given a pair of stereo images from both left and right camera it is possible to extract image depth.
Intensity of each pixel here corresponds to a distance from camera.

![new_depth](https://user-images.githubusercontent.com/61888740/77573548-e2345980-6ee1-11ea-8346-1b154917bdc9.png)

Example of estimated vehicle trajectory computed on a series of subsequent road images:

![traj](https://user-images.githubusercontent.com/61888740/77574579-4a376f80-6ee3-11ea-9e9a-549a6eb1420f.png)
