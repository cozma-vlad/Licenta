import pandas as pd
import sklearn as skl
import os
import pathlib as pl
import numpy as np
import cv2
import liveness_impl as lv

from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

live = {0:'fake', 1:'live'}
df = pd.read_csv("lbp_multiscale_features_ext_rose.csv")


y = df.pop('live').to_numpy()
x = df.to_numpy()[:, 1:]

(trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.25)


clf = svm.NuSVC(nu=0.387, gamma='scale')  # 0.387 -> "nu" optim
clf.fit(trainX, trainY)

dump(clf, 'model.joblib')


Y = clf.predict(testX)



acc = accuracy_score(testY, Y)
cm = confusion_matrix(testY, Y)

print(acc)
print(cm)