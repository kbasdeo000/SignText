# Code to capture the webcame input and process it.

import cv2
import numpy

# VideoCapture object:
# Argument = 0 because we are passing the webcam as input
cam = cv2.VideoCapture(0)

# Error checking:
if (cam.isOpened() == False):
    print("Unable to open video stream.")
    exit()

# Initializing blue box rectangle box data:
start = (30, 70)        # Start coordinates
end = (280, 400)        # End coordinates
color = (255, 0, 0)     # Blue in BGR
thickness = 3           # Line thickness

# Loop while the webcam is on/capturing data
while(cam.isOpened()):
    # Read the video and capture frame by frame:
    retval, frame = cam.read()

    # Draw a rectangle onto the screen:
    rectangle = cv2.rectangle(frame, start, end, color, thickness)

    #if retval == True:
        # Display the image:
        #cv2.imshow('Video Feed', frame)

    cv2.imwrite

    # Quits when pressing ESC
    if cv2.waitKey(1) == 27:
        break

# Release webcam:
cam.release()
# Close the window that shows the camera frames:
cv2.destoryAllWindows()
