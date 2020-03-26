# Code to capture the webcam input and process it.

import os
import shutil
import cv2
import numpy


# Draws a rectangle on the screen for the user to place their hand & make signs.
def draw_rectangle(frame):

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
def frames_directory():
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


# Main driver function
def main():
    # VideoCapture object:
    # Argument = 0 because we are passing the webcam as input
    cam = cv2.VideoCapture(0)

    # Error checking:
    if cam.isOpened() is False:
        print("Unable to open video stream.")
        exit()

    i = 0
    out_path = frames_directory()

    # Loop while the webcam is on/capturing data:
    while cam.isOpened():
        # Read the video and capture frame by frame:
        ret, frame = cam.read()

        draw_rectangle(frame)

        if ret is False:
            break

        # Flip the frame:
        # frame = cv2.flip(frame, 1)

        # Display the image:
        cv2.imshow('Video Feed', frame)

        # Saves only the rectangular part of the frame (minus the rectangular border)
        subframe = frame[72:397, 33:277]

        # Write the flipped frame to directory:
        cv2.imwrite(os.path.join(out_path, 'frame'+str(i)+'.jpg'), subframe)
        i += 1

        # Quits when pressing ESC key
        if cv2.waitKey(1) == 27:
            break

    # Release webcam:
    cam.release()
    # Close the window that shows the camera frames:
    cv2.destoryAllWindows()


# Driver code
main()
