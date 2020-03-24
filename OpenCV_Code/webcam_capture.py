# Code to capture the webcam input and process it.

import os
import shutil
import cv2
import numpy

# Checks if there is already an output folder for the frames
# If yes, it deletes the directory, then creates a new output folder
# If no, creates a new output directory
def frames_directory():
    cur_path = os.getcwd()  # Get the current working directory
    frm_path = '/Frames'
    out_path = cur_path + frm_path  # New path to write the frames

    # IF the new directory already exists, clear it by removing it
    if os.path.isdir(out_path):
        try:
            shutil.rmtree(out_path)
        except OSError:
            print("Error: Directory exists & cannot be removed.")

    try:
        os.mkdir(out_path)
        return out_path
    except OSError:
        print("Error: Creation of the 'frames' directory failed.")

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

i = 0
out_path = frames_directory()
# Loop while the webcam is on/capturing data:
while cam.isOpened():
    # Read the video and capture frame by frame:
    retval, frame = cam.read()

    # Draw a rectangle onto the screen:
    rectangle = cv2.rectangle(frame, start, end, color, thickness)

    if retval == False:
        break

    # Display the image:
    cv2.imshow('Video Feed', frame)

    # Flip the frame:
    #frame = cv2.flip(frame, 0)
    # Write the flipped frame:
    #out.write(frame)
    # out_path = '/home/kchonka/Documents/SignText/OpenCV_Code/Frames'
    #cv2.imwrite('frame'+str(i)+'.jpg',frame)
    cv2.imwrite(os.path.join(out_path, 'frame'+str(i)+'.jpg'), frame)
    i+=1


    # Quits when pressing ESC
    if cv2.waitKey(1) == 27:
        break

# Release webcam:
cam.release()
# Close the window that shows the camera frames:
cv2.destoryAllWindows()
