from pathlib import Path

import numpy as np
import pytest

# bytetracker
from bytetracker import BYTETracker

# static paths
OUTPUT_FOLDER = Path("tests/output")
TEST_INPUT_FOLDER = Path("tests/test_input")
EXPECTED_OUTPUT_FOLDER = Path("tests/expected_output")
OUTPUT_FOLDER = Path("tests/output")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def reading_detections_file(video_number):
    """Read the detections object file from a specific format

    Args:
        video_number (string): string representing the video name in folder

    Returns:
        df_detection (ndarray): A list of array of the tuples.
    """
    df_detection = np.loadtxt(
        TEST_INPUT_FOLDER / f"objects_detected_{video_number}.txt", delimiter=" "
    )
    print(df_detection)

    return df_detection


def reading_expected_results_from_txt(video_number):
    """Read the detections and tracked objects frames from Yolo and Bytetrack models

    Args:
        video_number (str): Video title to import

    Returns:
        expected_results_df (ndarray): array of frames detection and tracking imported from expected results.
    """
    expected_results_df = np.loadtxt(
        EXPECTED_OUTPUT_FOLDER / f"objects_detected_and_tracked_{video_number}.txt"
    )

    return expected_results_df


@pytest.mark.parametrize(
    "expected_results, test_input, video_number",
    [
        (
            reading_expected_results_from_txt("video1"),
            reading_detections_file("video1"),
            "video1",
        ),
        (
            reading_expected_results_from_txt("video2"),
            reading_detections_file("video2"),
            "video2",
        ),
    ],
)
def test_video_prediction_tracking(expected_results, test_input, video_number):
    """Execute the integration test: performing the tracking of test_input,
    whose output is to compare with expected_results

    Args:
        expected_result (DataFrame): the dataframe of the results expected to match the test result
        test_input (array): dataframe that is the import of prediction frames in test_input folder
        video_number (string): string component specific to a video

    Returns:
        Test assertion results
    """
    tracker = BYTETracker(track_thresh=0.15, track_buffer=3, match_thresh=0.85, frame_rate=12)
    test_results = []
    # first column of test input is the frame id
    frame_idx = np.unique(test_input[:, 0])
    test_results = []

    for frame_id in frame_idx:
        detections = test_input[test_input[:, 0] == frame_id][:, 1:]
        tracked_objects = tracker.update(detections, frame_id)
        if len(tracked_objects) > 0:
            tracked_objects = np.insert(
                tracked_objects, 0, np.full(len(tracked_objects), frame_id), axis=1
            )
            test_results.append(tracked_objects)

    if test_results:  # Ensure test_results is not empty to avoid errors
        combined_array = np.concatenate(test_results)
    else:
        combined_array = np.array([])  # Or handle the empty case as needed

    output_file_path = OUTPUT_FOLDER / f"{video_number}_test_results.txt"
    np.savetxt(output_file_path, combined_array, delimiter=" ", fmt="%s")

    np.array_equal(expected_results, combined_array)

    # Remove the file if the test is successful
    output_file_path.unlink()
