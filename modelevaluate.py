#!/usr/bin/env python3
# -*- coding = utf-8
import os
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from data.load_data import get_plant_leaves_dataset

# Get dataset images.
X_train, _, X_test, y_train, _, y_test = get_plant_leaves_dataset()

# Load and set up model.
model_path = os.path.join(os.path.dirname(__file__), 'data/model/Model-87-0.9622.hdf5')
model = load_model(model_path)

model.compile(
   optimizer = Adam(),
   loss = categorical_crossentropy,
   metrics = ['acc']
)

# Evaluate Model
print(model.evaluate(X_test, y_test))

# Create Confusion Matrix
predictions = list(np.argmax(item) for item in model.predict(X_test))
actual = list(np.argmax(item) for item in y_test)
cf = confusion_matrix(predictions, actual)
svm = sns.heatmap(cf, annot = True)
plt.show()

save = False
if save:
   fig = svm.get_figure()
   fig.savefig("confusionmatrix.png")

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(actual, predictions)
print("Matthews Correlation Coefficient Calculation: " + str(mcc))






