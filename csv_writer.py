import pandas as pd
import liveness_impl as lv
import os
import numpy as np
import cv2
import pathlib as pl
from blessings import Terminal
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
	help="path to output file")
args = vars(ap.parse_args())

FEATURE_LENGTH = 1364

cnt = 0
real_len = 0
fake_len = 0

for (root, _, fns) in os.walk(pl.Path(args['dataset'] + os.path.sep + "real")):
	real_len += len(fns)

for (root, _, fns) in os.walk(pl.Path(args['dataset'] + os.path.sep + "fake")):
	fake_len += len(fns)


res = np.empty((real_len + fake_len, FEATURE_LENGTH + 1), dtype=np.uint32)

for (root, _, fns) in os.walk(pl.Path(args['dataset'] + os.path.sep + "real")):
	real_len = len(fns)
	for fn in fns:
		im = cv2.imread(str(pl.Path(root).joinpath(fn)))
		print('\rreal ' + str(cnt + 1) + ' of ' + str(real_len), end="")
		#res = np.vstack((res, np.append(lv.feature_extractor_extended(im), np.array([1], dtype=np.uint32))))

		res[cnt, :FEATURE_LENGTH] = lv.feature_extractor_extended(im)
		res[cnt, FEATURE_LENGTH] = 1

		cnt += 1
#cnt = 0
print('')

for (root, _, fns) in os.walk(pl.Path(args['dataset'] + os.path.sep + "fake")):
	fake_len = len(fns)
	for fn in fns:
		im = cv2.imread(str(pl.Path(root).joinpath(fn)))
		print('\rfake ' + str(cnt - real_len + 1) + ' of ' + str(fake_len), end="")
		#res = np.vstack((res, np.append(lv.feature_extractor_extended(im), np.array([0], dtype=np.uint32))))

		res[cnt, :FEATURE_LENGTH] = lv.feature_extractor_extended(im)
		res[cnt, FEATURE_LENGTH] = 0

		cnt += 1

print('')
cols = ['lbp_feat_' + str(i) for i in range(1364)]
cols.append('live')

df = pd.DataFrame(res, columns=cols)
df.to_csv(args['output'])
print("Written " + args['output'])