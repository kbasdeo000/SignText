# Code to capture the webcame input and process it.

import cv2
import numpy

# VideoCapture object:
# Argument = 0 because we are passing the webcam as input
webcam_capture = cv2.VideoCapture(0)

# Error checking:
if (webcam_capture.isOpened() == False):
    print("Unable to open video stream.")
    exit()

# Initializing blue box rectangle box data:
start = (30, 70)         # Start coordinates
end = (250, 400)         # End coordinates
color = (255, 0, 0)     # Blue in BGR
thickness = 2           # Line thickness

# Loop while the webcam is on/capturing data
while(webcam_capture.isOpened()):
    # Read the video and capture frame by frame:
    ret, frame = webcam_capture.read()

    # Draw a rectangle onto the screen:
    rectangle = cv2.rectangle(frame, start, end, color, thickness)
    #cv2.imshow('VideoFeed', rectangle)

    if ret == True:
        # Display the image:
        cv2.imshow('Video Feed', frame)

    # Quits when pressing ESC
    if cv2.waitKey(1) == 27:
        break

# Release the video once we no longer need to access it:
webcam_capture.release()
# Close the window that shows the camera frames:
cv2.destoryAllWindows()
