from perception.src.core.stereo_depth_estimation_task import \
    StereoDepthEstimationTask


def test_stereo_depth_estimation_task():

    test_image_data_path = "image_data/*"
    num_disparities = 6 * 16
    block_size = 11
    test_task = StereoDepthEstimationTask(
        image_data_path=test_image_data_path,
        num_disparities=num_disparities,
        block_size=block_size,
    )
    result = test_task.execute()
    assert isinstance(result, dict)
    assert len(result) == 3
