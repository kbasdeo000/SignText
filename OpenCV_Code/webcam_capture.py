# Code to capture the webcam input and process it.

import os
import shutil
import cv2
import numpy


# This class accesses the webcam, captures frames and isolated the user's hand for ASL signs.
class WebcamCapture:
    def __init__(self):
        # VideoCapture object:
        # Argument = 0 because we are passing the webcam as input
        self.cam = cv2.VideoCapture(0)
        self.frames = []        # List of frame objects
        self.ready = True       # Represents webcam readiness

    # Performs error checking to check if the webcam is ready for capture/processing.
    # Returns true if yes, false if no.
    def is_ready(self):
        # Error checking:
        if self.cam.isOpened() is False:
            print("Unable to open video stream.")
            self.ready = False

        return self.ready

    # Draws a rectangle on the screen for the user to place their hand & make signs.
    def draw_rectangle(self, frame):

        # Initializing blue box rectangle box data:
        start = (30, 70)  # Start coordinates
        end = (280, 400)  # End coordinates
        color = (255, 0, 0)  # Blue in BGR
        thickness = 3  # Line thickness

        # Draw a rectangle onto the screen:
        cv2.rectangle(frame, start, end, color, thickness)

    # Checks if there is already an output folder for the frames
    # If yes, it deletes the directory, then creates a new output folder
    # If no, creates a new output directory
    def frames_directory(self):
        cur_path = os.getcwd()  # Get the current working directory
        fra_path = '/Frames'
        new_path = cur_path + fra_path  # New path to write the frames

        # IF the new directory already exists, clear it by removing it
        if os.path.isdir(new_path):
            try:
                shutil.rmtree(new_path)
            except OSError:
                print("Error: Directory exists & cannot be removed.")

        try:
            os.mkdir(new_path)
            return new_path
        except OSError:
            print("Error: Creation of the 'frames' directory failed.")

    # Function with main functionality: captures, displays and processes frame.
    def capture_frames(self):
        # Loop while the webcam is on/capturing data:
        while self.cam.isOpened():
            # Read the video and capture frame by frame:
            ret, frame = self.cam.read()

            self.draw_rectangle(frame)

            if ret is False:
                break

            # Flip the frame:
            # frame = cv2.flip(frame, 1)

            # Display the image:
            cv2.imshow('Video Feed', frame)

            # Saves only the rectangular part of the frame (minus the rectangular border)
            subframe = frame[72:397, 33:277]

            # Write the frames to the list:
            self.frames.append(subframe)

            # Quits when pressing ESC key
            if cv2.waitKey(1) == 27:
                break

        # Release webcam:
        self.cam.release()
        # Close the window that shows the camera frames:
        cv2.destoryAllWindows()

        return self.frames

