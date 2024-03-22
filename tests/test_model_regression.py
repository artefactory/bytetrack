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
    frame_id = None
    detections = []
    file_path = Path(f"test_input/objects_detected_{video_number}.txt")
    with file_path.open("r") as file:
        for line in file:
            parts = line.strip().split()
            frame_id_new = int(parts[0])
            if frame_id_new != frame_id:
                if frame_id is not None:
                    all_detections_by_frame.append((frame_id, np.array(detections)))
                    detections = []
                frame_id = frame_id_new
            detection = list(map(float, parts[1:]))
            detections.append(detection)
        if frame_id is not None and detections:
            all_detections_by_frame.append((frame_id, np.array(detections)))
    return all_detections_by_frame


def reading_expected_results_from_txt(video_number):
    """Read the detections and tracked objects frames from Yolo and Bytetrack models

    Args:
        Video title to import.

    Returns:
        cleaned dataframe consisting of concatenate object tracked frames.
    """
    expected_result_path = f"expected_output/objects_detected_and_tracked_{video_number}.txt"
    expected_results_df = pd.read_csv(expected_result_path, sep=" ", header=None)
    columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
    expected_results_df.columns = columns
    expected_results_df = expected_results_df.astype(float)
    return expected_results_df


@pytest.mark.parametrize(
    "expected_results, test_input",
    [
        (
            reading_expected_results_from_txt("video1"),
            read_detections_file("video1"),
        ),
    ],
)
def test_video_prediction_tracking(expected_results, test_input, byte_tracker):
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
    test_results_df = pd.DataFrame(combined_array, columns=["0", "1", "2", "3", "4", "5", "6", "7"])
    test_results_df = test_results_df.astype(float)

    pd.testing.assert_frame_equal(expected_results, test_results_df)
