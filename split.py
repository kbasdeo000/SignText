# Code to split the dataset 9/1 for test/validation

import split_folders

# Replace with the path of your local 'data_raw' directory
input_folder = '/home/kchonka/Documents/SignText/data_raw'

# Split with a ratio:
# We're only spliting into training/validation 9/1, so ratio=(.9, .1).
# Output is going to go into a folder named "output"
# This split distribution can be recreated by using the same seed #
split_folders.ratio(input_folder, output="data_split", seed=1337, ratio=(.9, .1))
