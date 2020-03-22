import os
from tempfile import TemporaryDirectory

from src.core.depth_estimation.estimate_stereo_depth import EstimateStereoDepth


def test_stereo_depth_estimation_task():
    test_image_data_path_template = "/Users/i.boytsov/Projects/perception/tests/core/data/stereo_image_data/test_data_road_{}"
    num_disparities = 6 * 16
    block_size = 11
    with TemporaryDirectory() as temp_dir:
        test_task = EstimateStereoDepth(
            dest_dir=temp_dir,
            image_data_path_template=test_image_data_path_template,
            num_disparities=num_disparities,
            block_size=block_size,
        )
        test_task.execute()
        assert len(os.listdir(temp_dir)) == 1
        assert os.path.exists(os.path.join(temp_dir, "000000.npy"))
