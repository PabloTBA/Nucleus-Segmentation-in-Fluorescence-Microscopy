import os
import utils.dirtools

config_vars = {}

# The main folder
config_vars["root_directory"] = './dataset/'

# The specific subfolders the download script expects
config_vars["raw_images_dir"] = './dataset/raw_images/'
config_vars["raw_annotations_dir"] = './dataset/raw_annotations/'