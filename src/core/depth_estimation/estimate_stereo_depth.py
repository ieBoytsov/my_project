import os

import cv2
import numpy as np

from src.core.utils.extra_func import mkdir_if_missing


class EstimateStereoDepth:
    def __init__(self, image_data_path_template, dest_dir, num_disparities, block_size):
        super().__init__()
        self.image_data_path_template = image_data_path_template
        self.dest_dir = dest_dir
        self.num_disparities = num_disparities
        self.block_size = block_size

    left_calib_matrix = np.array(
        [[640.0, 0.0, 640.0, 2176.0], [0.0, 480.0, 480.0, 552.0], [0.0, 0.0, 1.0, 1.4]]
    )
    right_calib_matrix = np.array(
        [[640.0, 0.0, 640.0, 2176.0], [0.0, 480.0, 480.0, 792.0], [0.0, 0.0, 1.0, 1.4]]
    )

    def get_image_pair(self, img_name):
        left_img = cv2.imread(
            os.path.join(self.image_data_path_template.format("left"), img_name)
        )
        right_img = cv2.imread(
            os.path.join(self.image_data_path_template.format("right"), img_name)
        )
        return left_img, right_img

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
        mkdir_if_missing(self.dest_dir)

        for img_name in sorted(
            os.listdir(self.image_data_path_template.format("left"))
        ):
            left_img, right_img = self.get_image_pair(img_name)
            disparity = self.compute_disparity_map(left_img, right_img)
            depth_map = self.compute_depth_map(disparity, camera_left, t_left, t_right)
            np.save(
                os.path.join(self.dest_dir, img_name.replace("png", "npy")), depth_map
            )
