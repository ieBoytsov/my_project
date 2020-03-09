import glob
import os

import numpy as np
from cv2 import cv2


class StereoDepthEstimationTask:
    def __init__(self, image_data_path, num_disparities, block_size):
        super().__init__()
        self.image_data_path = image_data_path
        self.num_disparities = num_disparities
        self.block_size = block_size

    left_calib_matrix = np.array(
        [[640.0, 0.0, 640.0, 2176.0], [0.0, 480.0, 480.0, 552.0], [0.0, 0.0, 1.0, 1.4]]
    )
    right_calib_matrix = np.array(
        [[640.0, 0.0, 640.0, 2176.0], [0.0, 480.0, 480.0, 792.0], [0.0, 0.0, 1.0, 1.4]]
    )

    def get_single_image_pair(self, subdir):
        pair = {}
        for img_name in os.listdir(subdir):
            if "right" in img_name:
                pair["right"] = cv2.imread(os.path.join(subdir, img_name))
            elif "left" in img_name:
                pair["left"] = cv2.imread(os.path.join(subdir, img_name))
            else:
                raise Exception(
                    "incorrect filename, each subdir must consist of a pair of left and right images only"
                )
        return pair

    def get_image_pairs(self):
        image_pairs = {}
        for subdir in glob.glob(self.image_data_path):
            image_pairs[os.path.basename(subdir)] = self.get_single_image_pair(subdir)
        return image_pairs

    def compute_disparity_map(self, img_left, img_right):

        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        matcher = cv2.StereoBM_create(
            numDisparities=self.num_disparities, blockSize=self.block_size
        )
        return matcher.compute(img_left, img_right).astype(np.float32) / 16

    def decompose_projection_matrix(self, calibration_matrix):

        (
            camera_matrix,
            rotation_vector,
            translation_vector,
            _,
            _,
            _,
            _,
        ) = cv2.decomposeProjectionMatrix(calibration_matrix)
        translation_vector /= translation_vector[3]

        return camera_matrix, rotation_vector, translation_vector

    def compute_depth_map(
        self, disparity_map, camera_left, translation_left, translation_right
    ):

        # Get the focal length from the K matrix
        focal_length = camera_left[0, 0]

        # Get the distance between the cameras from the t matrices (baseline)
        baseline = translation_left[1] - translation_right[1]

        # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
        disparity_map[disparity_map == 0] = 0.1
        disparity_map[disparity_map == -1] = 0.1

        # Initialize the depth map to match the size of the disparity map
        depth_map = np.ones(disparity_map.shape, np.single)

        # Calculate the depths
        depth_map[:] = focal_length * baseline / disparity_map[:]

        return depth_map

    def execute(self):
        camera_left, r_left, t_left = self.decompose_projection_matrix(
            self.left_calib_matrix
        )
        camera_right, r_right, t_right = self.decompose_projection_matrix(
            self.right_calib_matrix
        )

        image_pairs = self.get_image_pairs()
        depth_dict = {}
        for img_pair_name, img_pair_dict in image_pairs.items():
            left_img, right_img = img_pair_dict["left"], img_pair_dict["right"]
            disparity = self.compute_disparity_map(left_img, right_img)
            depth_map = self.compute_depth_map(disparity, camera_left, t_left, t_right)
            depth_dict[img_pair_name] = depth_map
        return depth_dict
