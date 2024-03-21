import pytest
import glob
# YOLO and video packages 
from ultralytics import YOLO
from bytetracker import BYTETracker
import cv2
import os
import subprocess
import numpy as np
import pandas as pd

# Fixture to load YOLO and BYTETracker models
@pytest.fixture
def model_creation():
    MODEL_WEIGHTS = "yolov8m.pt"
    yolo_model = YOLO(MODEL_WEIGHTS, task="detect")
    byte_tracker = BYTETracker(track_thresh=0.15, track_buffer=3, match_thresh=0.85, frame_rate=12)
    return yolo_model, byte_tracker

def extract_frames_from_video(video_path, video_number, fps=12):
    # enables running command in python file
    ffmpeg_cmd = f"ffmpeg -i {video_path} -vf fps={fps} frames/video_{video_number}_%d.png -hide_banner -loglevel panic"
    
    # Execute the command
    subprocess.run(ffmpeg_cmd, shell=True, check=True)

def read_text_to_dataframe(file_path):
    # Initialize an empty list to store the arrays
    arrays = []

    # Read the text file line by line
    with open(file_path, "r") as file:
        current_array = []
        for line in file:
            if line.strip():  # Check if the line is not empty
                row = [float(value) for value in line.strip().split("\t")]
                current_array.append(row)
            else:  # If a blank line is encountered, store the current array and reset for the next array
                arrays.append(current_array)
                current_array = []

    # If there's a remaining array after reading all lines, append it
    if current_array:
        arrays.append(current_array)

    # Convert the list of arrays into a list of numpy arrays
    numpy_arrays = [np.array(array) for array in arrays]

    # Convert the list of numpy arrays into a list of DataFrames
    dataframes = [pd.DataFrame(array) for array in numpy_arrays]

    return dataframes

def test_video_prediction_tracking(setup_models, video_path):
    yolo_model, byte_tracker = model_creation()
    
    frame_path = "frames/"
    os.makedirs(frame_path, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -vf fps=12 {frame_path}/frame_%d.png -hide_banner -loglevel error")
    frames = sorted(glob.glob(f"{frame_path}/*.png"))
    
    all_tracked_objects = []
    for frame_id, frame in enumerate(frames):
        img = cv2.imread(frame)
        # Assuming model and tracker have predict and track methods respectively
        predictions = yolo_model.predict(img)
        tracked_objects = byte_tracker.track(predictions)
        all_tracked_objects.append(tracked_objects)
    
    # Assert the equality of predicted and expected results
    # Note: You'll need to define a way to load or specify expected results.
    expected_results = read_text_to_dataframe(video_path)  # Implement this function

    assert all_tracked_objects == expected_results, "Tracked objects do not match expected results."
    
    # Optionally, clean up frames after test
    for frame in frames:
        os.remove(frame)
