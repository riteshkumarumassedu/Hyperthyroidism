
from __future__ import print_function
import os, sys
import numpy as np
import glob
from PIL import Image
import csv


class load_data():
    """
        get image data method will take the folder dir from where it is supposed to read the image files
        and returns the ndarray
    """
    def __init__(self):
        pass
    
def get_image_data_array(file_dir,image_labels_file):

    # first load the labels so that we can prepare the labels array

    labels_dict=get_labels_dict(image_labels_file)
    filelist = os.listdir(file_dir)
    
    labels=[]
    updated_file_list=[]
    
    # get the filename from the complete path
    for file,label in labels_dict.items():
        if file in filelist:
            labels.append(label)
            updated_file_list.append(file_dir+file)

    images_array = np.array([np.array(Image.open(fname)) for fname in updated_file_list])
    return images_array, labels


def get_labels_dict(labels_csv):
        labels_dict={}
        with open(labels_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:  # ignore the csv header
                    line_count += 1
                else:
                    labels_dict[row[0]+".jpg"]= row[1]
                    line_count += 1
        return labels_dict
    