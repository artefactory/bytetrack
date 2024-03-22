from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# bytetracker
from bytetracker import BYTETracker


@pytest.fixture
def byte_tracker():
    byte_tracker = BYTETracker(track_thresh=0.15, track_buffer=3, match_thresh=0.85, frame_rate=12)
    return byte_tracker


def read_detections_file(video_number):
    """Read the detections object file from a specific format

    Args:
        detections: A list of tuples composed of frame_id and its YOLO detections.

    Returns:
        A list of array of the tuples.
    """
    all_detections_by_frame = []

    # Construct file path
    file_path = f"test_input/objects_detected_{video_number}.txt"

    # Read file using pandas
    df_detection = pd.read_csv(file_path, sep=" ", header=None)

    # Group by frame_id and convert detections to numpy array
    grouped = df_detection.groupby(0)
    for frame_id, group in grouped:
        detections = group.iloc[:, 1:].to_numpy()
        all_detections_by_frame.append((frame_id, detections))

    return all_detections_by_frame


def reading_expected_results_from_txt(video_number):
    """Read the detections and tracked objects frames from Yolo and Bytetrack models

    Args:
        Video title to import.

    Returns:
        cleaned dataframe consisting of concatenated object tracked frames.
    """
    expected_result_path = f"expected_output/objects_detected_and_tracked_{video_number}.txt"
    expected_results_df = pd.read_csv(expected_result_path, sep=" ", header=None)

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
    tracker = byte_tracker
    test_results = []

    # reading the detected objects through Yolo model and apply tracking
    for frame_id, detections_bytetrack_format in test_input:
        tracked_objects = tracker.update(detections_bytetrack_format, frame_id)
        if len(tracked_objects) > 0:
            tracked_objects = np.insert(tracked_objects, 0, frame_id, axis=1)
            test_results.append(tracked_objects)
    # Cleaning the dataframe of tracked objects to align with expected output setup
    combined_array = np.concatenate(test_results)
    test_results_df = pd.DataFrame(combined_array)

    output_file_path = f"output/{video_number}_test_results.txt"
    test_results_df.to_csv(output_file_path, sep=" ", index=False, header=False)

    np.array_equal(expected_results.to_numpy(), test_results_df.to_numpy())

    # Remove the file if the test is successful
    Path.unlink(output_file_path)
