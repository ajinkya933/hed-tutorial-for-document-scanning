#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import cv2
import imghdr
import csv

from tensorflow import flags
flags.DEFINE_string('dataset_root_dir', '', 'Root directory to put all the training data.')
flags.DEFINE_string('dataset_folder_name', '', 'Folder name for training data. Can contain sub folder name.')
FLAGS = flags.FLAGS

if FLAGS.dataset_root_dir == '':
    print('must set dataset_root_dir')
    exit()
if FLAGS.dataset_folder_name == '':
    print('must set dataset_folder_name')
    exit()



'''
example:

python preprocess_generate_training_dataset.py --dataset_root_dir dataset \
                                               --dataset_folder_name generate_sample_by_ios_image_size_256_256_thickness_0.2
'''

if __name__ == "__main__":
    image_dataset_dir = os.path.join(FLAGS.dataset_root_dir, FLAGS.dataset_folder_name)
    if not os.path.exists(image_dataset_dir):
        print('path: {} is not exist, please check your data'.format(image_dataset_dir))

    csv_file_path = os.path.join(FLAGS.dataset_root_dir, '{}.csv'.format(FLAGS.dataset_folder_name))

    f = open(csv_file_path, 'wb')
    csv_writer = csv.writer(f)

    # !! When TensorFlow reads csv, it does not need or process the line title information, only the data is read directly.


    # csv_writer.writerow(['image_name', 'annotation_image_name']) 

    images = []
    path_prefix = FLAGS.dataset_folder_name + '/'

    for root, dirs, files in os.walk(image_dataset_dir):
        for name in files:
            filepath = os.path.join(root, name)
            # print('file path is:{}'.format(filepath))
            # http://stackoverflow.com/questions/889333/how-to-check-if-a-file-is-a-valid-image-file
            what = imghdr.what(filepath)
            if what == 'jpeg' or what == 'png':

                '''
                # Using the following method to detect the validity of image is too slow, so I give up
                image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                # When running, I found that there are abnormal images in the collection, and I will encounter the following problems.


                if image is None:
                    print('image is None, path is: {}\n'.format(filepath))
                    continue

                if len(image.shape) != 3:
                    print('image.shape is: {},  path is: {}\n'.format(image.shape, filepath))
                    continue
                '''

                if name.endswith('.jpg'):
                    # Eligible picture
                    name = os.path.splitext(name)[0]
                    # print('get_images_in_path, filepath is: {}, file name is: {}'.format(filepath, name))
                    images.append(name[:-6])# Name is like: zZu7xlG8IE_random_size_19_32_1_color, need to remove the following '_color' field
                    # http://stackoverflow.com/questions/10645959/how-do-i-remove-the-last-n-characters-from-a-string

    print('total {} *_color.jpg images'.format(len(images)))

    for name in images:
        color_image_name = name + '_color.jpg'
        annotation_image_name = name + '_annotation.png'
        annotation_gray_image_name = name + '_annotation_gray.png'
        annotation_thresh_gray_image_name = name + '_annotation_thresh_gray.png'

        color_image_path = os.path.join(image_dataset_dir, color_image_name)
        annotation_image_path = os.path.join(image_dataset_dir, annotation_image_name)
        annotation_gray_image_path = os.path.join(image_dataset_dir, annotation_gray_image_name)
        annotation_thresh_gray_image_path = os.path.join(image_dataset_dir, annotation_thresh_gray_image_name)

        image = cv2.imread(annotation_image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print('annotation image is None, path is: {}'.format(annotation_image_path))
            continue

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        cv2.imwrite(annotation_gray_image_path, grayImage)

        #Binarization, this picture is the Y in the training sample.
        ret, threshGrayImage = cv2.threshold(grayImage, 20, 255, cv2.THRESH_BINARY)
        cv2.imwrite(annotation_thresh_gray_image_path, threshGrayImage)

        # Csv does not store absolute paths. In input_pipeline.py, it still needs to be assembled according to FLAGS.dataset_root_dir.
        csv_writer.writerow([FLAGS.dataset_folder_name + '/' + color_image_name, 
                             FLAGS.dataset_folder_name + '/' + annotation_thresh_gray_image_name])


    f.close()
    print('finished')
