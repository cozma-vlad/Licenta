import pandas as pd
import liveness_impl as lv
import os
import numpy as np
import cv2
import pathlib as pl

res = np.zeros((834, ), dtype=np.uint32) #init for vstacking

for (root, _, fns) in os.walk(pl.Path("H:/Lic/NormalizedFace").joinpath("ClientNormalized")):
	for fn in fns:
		im = cv2.imread(str(pl.Path(root).joinpath(fn)))

		if im is None:
			print(root + '\\' + fn)

		res = np.vstack((res, np.append(lv.feature_extractor(im), np.array([1], dtype=np.uint32))))

for (root, _, fns) in os.walk(pl.Path("H:/Lic/NormalizedFace").joinpath("ImposterNormalized")):
	for fn in fns:
		im = cv2.imread(str(pl.Path(root).joinpath(fn)))

		if im is None:
			print(root + '\\' + fn)
		res = np.vstack((res, np.append(lv.feature_extractor(im), np.array([0], dtype=np.uint32))))

cols = ['lbp_feat_' + str(i) for i in range(833)]
cols.append('live')

df = pd.DataFrame(res[1:], columns=cols)
df.to_csv("lbp_multiscale_features_nuaa.csv")