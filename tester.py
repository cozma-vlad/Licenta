from joblib import load
import liveness_impl as lv
import numpy as np
import os
import pathlib as pl
import cv2

clf = load('model.joblib')

train_idx   = np.load('train.npy')
test_idx    = np.load('test.npy')


train_idx_true = []
test_idx_true = []


cl_file = open("NormalizedFace/client_test_normalized.txt")
imp_file = open("NormalizedFace/imposter_test_normalized.txt")

contents = [str(pl.Path("ClientNormalized") / x.split('\\')[0] / x.split('\\')[1].strip()) for x in cl_file.readlines()]
contents += [str(pl.Path("ImposterNormalized") / x.split('\\')[0] / x.split('\\')[1].strip()) for x in imp_file.readlines()]

#contents.sort()

Y = np.zeros((9123, ), dtype=np.uint32)
Y[:3362] = 1

cnt = 0
correct = 0

for i in contents:
    im = cv2.imread('NormalizedFace/' + i)
    fv = lv.feature_extractor(im)
    y = clf.predict(fv.reshape(1, -1))

    if y[0] == Y[cnt]:
        correct += 1
    cnt += 1

#for root, dirs, fns in os.walk('NormalizedFace/ClientNormalized'):


