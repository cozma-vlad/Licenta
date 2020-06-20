import pandas as pd
import sklearn as skl
import os
import pathlib as pl
import numpy as np
import cv2
import liveness_impl as lv

from sklearn import svm


df = pd.read_csv("lbp_multiscale_features_nuaa.csv")


y = df.pop('live').to_numpy()
x = df.to_numpy()

train_idx   = np.load('train.npy')
test_idx    = np.load('test.npy')


clf = svm.NuSVC(nu=0.387, gamma='scale')  # 0.387 -> "nu" optim
	#print(x[:, 1:], x.shape)
clf.fit(x[train_idx, 1:], y[train_idx])

Y = clf.predict(x[test_idx, 1:])
acc = np.where(Y == y[test_idx])[0].shape[0] / Y.shape[0] * 100

print(acc)

cap = cv2.VideoCapture('face2.mp4')
clf_face_det = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')


live = {0:'fake', 1:'live'}

while True:
	ret, im = cap.read()
	if not ret:
		cap = cv2.VideoCapture('face.mp4')
	if im is None:
		continue

	faces = clf_face_det.detectMultiScale(im, scaleFactor=1.3, minSize=(30, 30))

	for _, (x, y, w, h) in enumerate(faces):
		fv = lv.feature_extractor(im)
		res = clf.predict(fv.reshape(1, -1))

		cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0))
		cv2.putText(im, live[res[0]], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=2)
		break

	c = cv2.waitKey(20)
	if c==27:
		cv2.destroyAllWindows()
		break
	cv2.imshow("img", im)