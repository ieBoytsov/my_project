import argparse
import errno
import os
from typing import List, Tuple

import cv2
import numpy as np


def mkdir_if_missing(dir_path: str):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class EstimateVehicleTrajectory:
    """
    this class estimates trajectory of vehicle from a series of images. it ectracts and matches features
    in order to estimation motion by exploring subsequent images
    ...

    Attributes
    ----------
    image_data_dir : str
        path to subsequent images
    dest_dir : str
        path to save output trajectory
    dist_threshold: int
        maximum allowed relative distance between the best matches, (0.0, 1.0)
    do_filter:
        whether or not to filter imprecise matches by distance_threshold
    http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    """

    def __init__(
        self, image_data_dir: str, dest_dir: str, dist_threshold: float, do_filter: bool
    ):
        super().__init__()
        self.image_data_dir = image_data_dir
        self.dest_dir = dest_dir
        self.dist_threshold = dist_threshold
        self.do_filter = do_filter
        self.camera_matrix = np.array(
            [[640, 0, 640], [0, 480, 480], [0, 0, 1]], dtype=np.float32
        )

    def extract_features(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:

        # Initiate ORB detector
        orb = cv2.ORB_create(3000)

        kp, des = orb.detectAndCompute(image, None)

        return kp, des

    def match_features(
        self, des1: np.ndarray, des2: np.ndarray
    ) -> List[List[cv2.DMatch]]:
        """
        Match features from two images using keypoint descriptors of a pair of images
        match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
        """
        bf = cv2.BFMatcher_create()
        match = bf.knnMatch(des1, des2, k=2)

        return match

    def filter_matches_distance(
        self, match: List[List[cv2.DMatch]]
    ) -> List[List[cv2.DMatch]]:
        """
        Filter matched features from two images by distance between the best matches

        Arguments:
        match -- list of matched features from two images
        dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

        Returns:
        filtered_match -- list of good matches, satisfying the distance threshold
        """

        filtered_match = []
        for m, n in match:
            if m.distance < self.dist_threshold * n.distance:
                filtered_match.append([m])

        return filtered_match

    def estimate_motion(
        self,
        match: List[List[cv2.DMatch]],
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
    ) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[List[float]]]:
        """
        Estimate camera motion from a pair of subsequent image frames

        Arguments:
        match -- list of matched features from the pair of images
        kp1 -- list of the keypoints in the first image
        kp2 -- list of the keypoints in the second image

        Optional arguments:
        depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

        Returns:
        R -- recovered 3x3 rotation numpy matrix
        t -- recovered 3x1 translation numpy vector
        image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are
                         coordinates of the i-th match in the image coordinate system
        image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are
                         coordinates of the i-th match in the image coordinate system

        """

        image1_points, image2_points = [], []

        for m in match:
            m = m[0]
            query_idx = m.queryIdx
            train_idx = m.trainIdx

            # get first img matched keypoints
            p1_x, p1_y = kp1[query_idx].pt
            image2_points.append([p1_x, p1_y])

            # get second img matched keypoints
            p2_x, p2_y = kp2[train_idx].pt
            image1_points.append([p2_x, p2_y])

        # essential matrix
        E, mask = cv2.findEssentialMat(
            np.array(image1_points), np.array(image2_points), self.camera_matrix
        )
        _, R, t, mask = cv2.recoverPose(
            E, np.array(image1_points), np.array(image2_points), self.camera_matrix
        )

        return R, t, image1_points, image2_points

    def estimate_trajectory(
        self, matches: List[List[List[cv2.DMatch]]], kp_list: List[List[cv2.KeyPoint]]
    ) -> np.ndarray:
        """
        Estimate complete camera trajectory from subsequent image pairs

        Arguments:
        matches -- list of matches for each subsequent image pair in the dataset.
                   Each matches[i] is a list of matched features from images i and i + 1
        kp_list -- a list of keypoints for each image in the dataset

        Returns:
        trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and
                      trajectory[:, i] is a 3x1 numpy vector, such as:

                      trajectory[:, i][0] - is X coordinate of the i-th location
                      trajectory[:, i][1] - is Y coordinate of the i-th location
                      trajectory[:, i][2] - is Z coordinate of the i-th location

                      * Consider that the origin of your trajectory cordinate system is located at the camera position
                      when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven
                      at the initialization of this function

        """

        trajectory = [np.array([0, 0, 0])]
        P = np.eye(4)

        for i in range(len(matches)):
            match = matches[i]
            kp1 = kp_list[i]
            kp2 = kp_list[i + 1]
            rmat, tvec, image1_points, image2_points = self.estimate_motion(
                match, kp1, kp2
            )
            R = rmat
            t = np.array([tvec[0, 0], tvec[1, 0], tvec[2, 0]])

            P_new = np.eye(4)
            P_new[0:3, 0:3] = R.T
            P_new[0:3, 3] = (-R.T).dot(t)
            P = P.dot(P_new)

            trajectory.append(P[:3, 3])

        trajectory = np.array(trajectory).T
        trajectory[2, :] = -1 * trajectory[2, :]

        return trajectory

    def execute(self):
        # load subsequent images and extract keypoints and feature descriptors
        kp_list = []
        des_list = []
        for img_name in sorted(os.listdir(self.image_data_dir)):
            img = cv2.imread(os.path.join(self.image_data_dir, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kp, des = self.extract_features(img)
            kp_list.append(kp)
            des_list.append(des)

        # match features
        matches = []

        for i in range(len(des_list) - 1):
            des1 = des_list[i]
            des2 = des_list[i + 1]
            match = self.match_features(des1, des2)
            matches.append(match)

        # optionally filter matches by distance threshold
        if self.do_filter:
            filtered_matches = []

            for i in range(len(matches)):
                filtered_matches.append(self.filter_matches_distance(matches[i]))
            matches = filtered_matches

        # calculate vehicle trajectory from matches and keypoints
        trajectory = self.estimate_trajectory(matches, kp_list)
        mkdir_if_missing(self.dest_dir)
        np.save(os.path.join(self.dest_dir, "trajectory.npy"), trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vehicle trajectory estimator")

    parser.add_argument(
        "--image_data_dir", type=str, metavar="PATH", help="path to subsequent images",
    )
    parser.add_argument(
        "--dest_dir", type=str, metavar="PATH", help="path to save output trajectory"
    )
    parser.add_argument("--dist_threshold", type=float, default=0.6)
    parser.add_argument("--do_filter", type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)
    estimate_trajectory_task = EstimateVehicleTrajectory(**args)
    estimate_trajectory_task.execute()
