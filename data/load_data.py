#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import pickle

import numpy as np

def get_plant_leaves_dataset(savedir = None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
   # Returns the six core arrays which make up the processed plant leaves dataset.
   if savedir is None: savedir = os.path.join(os.path.dirname(__file__), 'dataset', 'plant_leaves')
   if not os.path.exists(os.path.join(savedir, 'X_train.pickle')):
      raise FileNotFoundError("One or more of the processed dataset pieces are missing. "
                              "Please preprocess the data before attempting to load it.")
   with open(os.path.join(savedir, 'X_train.pickle'), 'rb') as file:
      X_train = pickle.load(file)
   with open(os.path.join(savedir, 'X_validation.pickle'), 'rb') as file:
      X_validation = pickle.load(file)
   with open(os.path.join(savedir, 'X_test.pickle'), 'rb') as file:
      X_test = pickle.load(file)
   with open(os.path.join(savedir, 'y_train.pickle'), 'rb') as file:
      y_train = pickle.load(file)
   with open(os.path.join(savedir, 'y_validation.pickle'), 'rb') as file:
      y_validation = pickle.load(file)
   with open(os.path.join(savedir, 'y_test.pickle'), 'rb') as file:
      y_test = pickle.load(file)

   return X_train, X_validation, X_test, y_train, y_validation, y_test

