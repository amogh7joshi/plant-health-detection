#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import time
import pickle
import argparse

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Preprocessing Functions and Main Script for the Plant Leaves Dataset.

def make_set_dirs(image = False) -> None:
   # Make Train, Test, and Validation Directories w/ Healthy and Diseased Subdirectories.
   # Image parameter should be set to true when using image directories instead of np.ndarrays.
   # CURRENTLY NOT IMPLEMENTED, so do not change.
   if image:
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'train')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'train'))
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'train', 'healthy')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'train', 'healthy'))
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'train', 'diseased')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'train', 'diseased'))

      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'test')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'test'))
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'test', 'healthy')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'test', 'healthy'))
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'test', 'diseased')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'test', 'diseased'))

      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'validation')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'validation'))
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'validation', 'healthy')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'validation', 'healthy'))
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'validation', 'diseased')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'validation', 'diseased'))
   else:
      if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'dataset', 'plant_leaves')):
         os.mkdir(os.path.join(os.path.dirname(__file__), 'dataset', 'plant_leaves'))

def split_data(datadir, datalist) -> (list, list):
   # Split Images into Healthy/Diseased.
   healthy = []; diseased = []
   for set in datalist:
      health = os.listdir(os.path.join(datadir, set))
      for h in health:
         # Watch out for .DS_Store on MacOS.
         if sys.platform == 'darwin' and h == ".DS_Store": continue
         plants = os.listdir(os.path.join(datadir, set, h))
         for plant in plants:
            # Watch out for .DS_Store on MacOS.
            if sys.platform == 'darwin' and h == ".DS_Store": continue
            if h == 'diseased': diseased.append(os.path.join(datadir, set, h, plant))
            if h == 'healthy': healthy.append(os.path.join(datadir, set, h, plant))

   return healthy, diseased

def save_healthy(healthy) -> None:
   # Get all of the healthy images and place them into an array.
   healthy_imgs = np.empty((len(healthy), 256, 256, 3), dtype = np.uint8)
   for indx, image in enumerate(healthy):
      t = time.time()
      img = cv2.imread(image)
      x, y = img.shape[0], img.shape[1]
      pixels = cv2.resize(img[250:x - 250, 250: y - 250], (256, 256))
      healthy_imgs[indx, ...] = pixels
      print(indx, " - ", time.time() - t)

   # Save it to a pickle file because trying to conduct multiple operations will drain memory.
   # The pickle file will be used in a second run of the same script.
   with open("healthy.pickle", "wb") as file:
      pickle.dump(healthy_imgs, file)

def save_diseased(diseased) -> None:
   # Get all of the diseased images and place them into an array.
   diseased_imgs = np.empty((len(diseased), 256, 256, 3), dtype = np.uint8)
   for indx, image in enumerate(diseased):
      t = time.time()
      img = cv2.imread(image)
      x, y = img.shape[0], img.shape[1]
      pixels = cv2.resize(img[250:x - 250, 250: y - 250], (256, 256))
      diseased_imgs[indx, ...] = pixels
      print(indx, " - ", time.time() - t)

   # Save it to a pickle file because trying to conduct multiple operations will drain memory.
   # The pickle file will be used in a second run of the same script.
   with open("diseased.pickle", "wb") as file:
      pickle.dump(diseased_imgs, file)

def save_main_dataset(savedir) -> None:
   # Get diseased/healthy image arrays.
   with open('diseased.pickle', 'rb') as file:
      diseased = pickle.load(file)
   with open('healthy.pickle', 'rb') as file:
      healthy = pickle.load(file)

   # Create the total (diseased + healthy) array.
   tot = np.concatenate((diseased, healthy))

   # Create the labels array.
   labels = np.empty((tot.shape[0],))
   labels[0:diseased.shape[0]] = 0
   labels[diseased.shape[0]:diseased.shape[0] + healthy.shape[0]] = 1
   labels = labels.astype(np.int32)
   labels = to_categorical(labels)

   # Create the training, validation, and testing sets.
   X_train, X_temp, y_train, y_temp = train_test_split(tot, labels, shuffle = True, test_size = 0.2)
   X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, shuffle = True, test_size = 0.5)

   # Save dataset to binary pickle files.
   with open(os.path.join(savedir, 'X_train.pickle'), 'wb') as file:
      pickle.dump(X_train, file)
   with open(os.path.join(savedir, 'X_validation.pickle'), 'wb') as file:
      pickle.dump(X_validation, file)
   with open(os.path.join(savedir, 'X_test.pickle'), 'wb') as file:
      pickle.dump(X_test, file)
   with open(os.path.join(savedir, 'y_train.pickle'), 'wb') as file:
      pickle.dump(y_train, file)
   with open(os.path.join(savedir, 'y_validation.pickle'), 'wb') as file:
      pickle.dump(y_validation, file)
   with open(os.path.join(savedir, 'y_test.pickle'), 'wb') as file:
      pickle.dump(y_test, file)

if __name__ == '__main__':
   # Predetermined constants.
   datadir = os.path.join(os.path.dirname(__file__), 'hb74ynkjcn-1')
   savedir = os.path.join(os.path.dirname(__file__), 'dataset', 'plant_leaves')
   datalist = list(os.walk(datadir))[0][1]

   # Parse Command Line Argument(s) for Script.
   ap = argparse.ArgumentParser()
   ap.add_argument('-m', "--mode", default = None,
                   help = "The specific part of the script that you are trying to run.")
   args = vars(ap.parse_args())

   if args['mode'] is None:
      raise OSError("If not running from the script, then you must choose which functions to run manually.")

   # Different Modes of this Script.
   if args['mode'] == "split": # Split Everything
      print('Mode: Split')
      make_set_dirs() # Make directories.
      healthy, diseased = split_data(datadir, datalist) # Split all data into healthy/diseased.
      save_healthy(healthy) # Save the healthy images (in an np.ndarray) to a pickle file.
      save_diseased(diseased) # Save the diseased images (in an np.ndarray) to a pickle file.
   elif args['mode'] == "split-healthy": # Split only Healthy Images (to not drain system).
      print('Mode: Split-Healthy')
      make_set_dirs() # Make directories.
      healthy, diseased = split_data(datadir, datalist) # Split all data into healthy/diseased.
      save_healthy(healthy) # Save the healthy images (in an np.ndarray) to a pickle file.
   elif args['mode'] == "split-diseased": # Split only Diseased Images (to not drain system).
      print('Mode: Split-Healthy')
      make_set_dirs() # Make directories.
      healthy, diseased = split_data(datadir, datalist) # Split all data into healthy/diseased.
      save_diseased(diseased) # Save the healthy images (in an np.ndarray) to a pickle file.
   if args['mode'] == "save":  # Save the final processed dataset.
      print('Mode: Save') # Save the training/validation/test datasets to binary pickle files.
      save_main_dataset(savedir)

   print('Processing Complete.')





