import pandas as pd
import liveness_impl as lv
import os
import numpy as np
import cv2
import pathlib as pl
from blessings import Terminal


res = np.zeros((1365, ), dtype=np.uint32) #init for vstacking

cnt = 0
real_len = 0
fake_len = 0



for (root, _, fns) in os.walk(pl.Path("/home/uzzi/Downloads/rose_photos/real")):
	real_len = len(fns)
	for fn in fns:
		im = cv2.imread(str(pl.Path(root).joinpath(fn)))

		print('real ' + str(cnt) + ' of ' + str(real_len) + '\r', end="")
		cnt += 1
		res = np.vstack((res, np.append(lv.feature_extractor_extended(im), np.array([1], dtype=np.uint32))))

cnt = 0
print('\n')

for (root, _, fns) in os.walk(pl.Path("/home/uzzi/Downloads/rose_photos/fake")):
	fake_len = len(fns)
	for fn in fns:
		im = cv2.imread(str(pl.Path(root).joinpath(fn)))

		print('fake ' + str(cnt) + ' of ' + str(fake_len) + '\r', end="")
		cnt += 1
		res = np.vstack((res, np.append(lv.feature_extractor_extended(im), np.array([0], dtype=np.uint32))))

cols = ['lbp_feat_' + str(i) for i in range(1364)]
cols.append('live')

df = pd.DataFrame(res[1:], columns=cols)
df.to_csv("lbp_multiscale_features_ext_rose.csv")