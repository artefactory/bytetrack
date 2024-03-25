from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# bytetracker
from bytetracker import BYTETracker

# static paths
TEST_INPUT_PATH = "tests/test_input/objects_detected_"
EXPECTED_OUTPUT_PATH = "tests/expected_output/objects_detected_and_tracked_"
OUTPUT_PATH = "tests/output"
OUTPUT_FILE_SUFFIX = "_test_results.txt"


@pytest.fixture
def byte_tracker():
    """Generate byte tracker model instance

    Returns:
        ByteTracker object
    """
    byte_tracker = BYTETracker(track_thresh=0.15, track_buffer=3, match_thresh=0.85, frame_rate=12)
    return byte_tracker


def read_detections_file(video_number):
    """Read the detections object file from a specific format

    Args:
        detections: A list of tuples composed of frame_id and its YOLO detections.

    Returns:
        A list of array of the tuples.
    """
    df_detection = pd.read_csv(f"{TEST_INPUT_PATH}{video_number}.txt", sep=" ")

    return df_detection


def reading_expected_results_from_txt(video_number):
    """Read the detections and tracked objects frames from Yolo and Bytetrack models

    Args:
        Video title to import.

    Returns:
        cleaned dataframe consisting of concatenated object tracked frames.
    """
    expected_results_df = pd.read_csv(f"{EXPECTED_OUTPUT_PATH}{video_number}.txt", sep=" ")

    return expected_results_df


@pytest.mark.parametrize(
    "expected_results, test_input, video_number",
    [
        (
            reading_expected_results_from_txt("video1"),
            read_detections_file("video1"),
            "video1",
        ),
        (
            reading_expected_results_from_txt("video2"),
            read_detections_file("video2"),
            "video2",
        ),
    ],
)
def test_video_prediction_tracking(expected_results, test_input, video_number, byte_tracker):
    """Execute the integration test: performing the tracking of test_input,
    whose output is to compare with expected_results

    Args:
        expected_result: the dataframe of the results expected to match the test result
        test_input: dataframe that is the import of prediction frames in test_input folder
        video_number: string component specific to a video
        byte_tracker: bytetracker object set up by fixture

    Returns:
        Test assertion results
    """
    tracker = byte_tracker
    test_results = []

    # reading the detected objects through Yolo model and apply tracking
    frame_idx = test_input["frame_id"].unique()
    for frame_id in frame_idx:
        detections = test_input[test_input["frame_id"] == frame_id][
            ["x_min", "y_min", "x_max", "y_max", "confidence", "class_id"]
        ].to_numpy()
        tracked_objects = tracker.update(detections, frame_id)
        if len(tracked_objects) > 0:
            tracked_objects = np.insert(tracked_objects, 0, frame_id, axis=1)
            test_results.append(tracked_objects)
    # Cleaning the dataframe of tracked objects to align with expected output setup
    combined_array = np.concatenate(test_results)

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    output_file_path = f"{OUTPUT_PATH}/{video_number}{OUTPUT_FILE_SUFFIX}"
    np.savetxt(output_file_path, combined_array, delimiter=" ", fmt="%s")

    np.array_equal(expected_results.to_numpy(), combined_array)

    # Remove the file if the test is successful
    Path(output_file_path).unlink()
