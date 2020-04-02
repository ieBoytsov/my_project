import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

from src.core.utils.extra_func import mkdir_if_missing


class ComputeDistanceToCollision:
    """
    This class uses detected bounding boxes and corresponding depths to find the distance to the
    closest obstacle on the road for a given image

    ...

    Attributes
    ----------
    depth_data_dir : str
        path to images depth files
    masks_data_dir : str
        path to corresponding bbox annotations
    dest_dir : str
        path to save output info
    """

    def __init__(self, depth_data_dir: str, masks_data_dir: str, dest_dir: str):
        super().__init__()
        self.depth_data_dir = depth_data_dir
        self.masks_data_dir = masks_data_dir
        self.dest_dir = dest_dir
        self.file_names = [x.split(".")[0] for x in os.listdir(self.depth_data_dir)]

    def class_mapping(self, obj_name: str) -> int:
        """default mapping of labels"""

        classes = [
            "Car",
            "Van",
            "Truck",
            "Pedestrian",
            "Person_sitting",
            "Cyclist",
            "Tram",
            "Misc",
            "DontCare",
        ]
        mapping = {}
        for idx, obj in enumerate(classes):
            mapping[obj] = idx

        return mapping[obj_name]

    def load_bboxes(self, mask_dir: str) -> np.ndarray:
        """load bounding boxes with all detected objects on image"""

        with open(mask_dir) as f:
            content = f.readlines()
        content = [x.split() for x in content]
        content = [x for x in content if x[0] != "DontCare"]
        boxes = np.empty((len(content), 5), dtype=np.int)
        for idx, item in enumerate(content):
            y_min, x_min = int(float(item[4])), int(float(item[5]))
            y_max, x_max = int(float(item[6])), int(float(item[7]))

            boxes[idx, :] = [self.class_mapping(item[0]), x_min, y_min, x_max, y_max]

        return boxes

    def get_img_data(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """get corresponding boxes and depth map for single image"""

        boxes = self.load_bboxes(os.path.join(self.masks_data_dir, file_name + ".txt"))
        depth_map = np.load(os.path.join(self.depth_data_dir, file_name + ".npy"))
        return boxes, depth_map

    def get_distance_to_closest_object(
        self, detected_objects: np.ndarray, depth_map: np.ndarray
    ) -> Dict[str, str]:
        """compute distance to every obstacle at the image and return the closest one"""

        closest_point_depth = np.inf
        closest_object = None
        for bbox in detected_objects:
            detected_object, x_min, y_min, x_max, y_max = bbox
            obstacle_depth = depth_map[x_min:x_max, y_min:y_max]
            curr_point_depth = obstacle_depth.min()
            if curr_point_depth < closest_point_depth:
                closest_point_depth = curr_point_depth
                closest_object = detected_object
        return {str(closest_object): str(closest_point_depth)}

    def execute(self):
        mkdir_if_missing(self.dest_dir)
        for file_name in sorted(self.file_names):
            detected_objects, depth_map = self.get_img_data(file_name)
            object_distance_mapping = self.get_distance_to_closest_object(
                detected_objects, depth_map
            )
            with open(
                os.path.join(self.dest_dir, "{}.json".format(file_name)), "w"
            ) as fp:
                json.dump(object_distance_mapping, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute distance to collision")

    parser.add_argument(
        "--depth_data_dir", type=str, metavar="PATH", help="path to images depth files"
    )
    parser.add_argument(
        "--masks_data_dir",
        type=str,
        metavar="PATH",
        help="path to corresponding bbox annotations",
    )
    parser.add_argument(
        "--dest_dir", type=str, metavar="PATH", help="path to save output info"
    )
    args = parser.parse_args()
    args = vars(args)
    compute_distance_task = ComputeDistanceToCollision(**args)
    compute_distance_task.execute()
