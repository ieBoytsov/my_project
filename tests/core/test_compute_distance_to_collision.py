import json
import os
from tempfile import TemporaryDirectory

import pkg_resources

from src.core.obstacle_distance_estimation.compute_distance_to_collision import \
    ComputeDistanceToCollision


def test_compute_distance_to_collision():
    depth_path_template = "data/obstacle_location_data/depths"
    depth_data_dir = pkg_resources.resource_filename(__name__, depth_path_template)

    masks_path_template = "data/obstacle_location_data/masks"
    masks_data_dir = pkg_resources.resource_filename(__name__, masks_path_template)
    with TemporaryDirectory() as temp_dir:
        test_task = ComputeDistanceToCollision(
            dest_dir=temp_dir,
            depth_data_dir=depth_data_dir,
            masks_data_dir=masks_data_dir,
        )
        test_task.execute()

        with open(os.path.join(temp_dir, "000008.json")) as f:
            content = json.load(f)
            assert isinstance(content, dict)
            assert content == {"0": "3.368421"}
