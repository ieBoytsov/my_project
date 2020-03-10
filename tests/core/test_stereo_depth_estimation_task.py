from perception.src.core.stereo_depth_estimation_task import \
    StereoDepthEstimationTask


def test_stereo_depth_estimation_task():

    test_image_data_path_template = "image_data/test_data_road_{}"
    num_disparities = 6 * 16
    block_size = 11
    test_task = StereoDepthEstimationTask(
        image_data_path_template=test_image_data_path_template,
        num_disparities=num_disparities,
        block_size=block_size,
    )
    result = test_task.execute()
    assert isinstance(result, list)
