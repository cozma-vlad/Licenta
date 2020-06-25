import  cupy  as cp
import  numpy as np
import  cv2
import  os
from    pathlib import Path
from 	skimage import feature
from    numba import cuda, njit, prange, jit

d_types = cp.array([[8, 1], [8, 2], [16, 2]], dtype=cp.uint16)

bins8 	= np.array([], dtype=np.uint32)
bins16	= np.array([], dtype=np.uint32)


d_d = cp.array([ [ 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
			   [ 0, -2, 1, -1, 2, 0, 1, 1, 0, 2, -1, 1, -2, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
			   [ 0, -2, 1, -2, 1, -1, 2, -1, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2, -1, 2, -1, 1, -2, 1, -2, 0, -2, -1, -1, -1, -2, -1] 
			], dtype=cp.int8)

corners = [ (0, 0), (0, 21), (0, 38),
			(17, 0), (17, 21), (17, 38),
			(38, 0), (38, 21), (38, 38)]


@cuda.jit(device=True)
def lbp(res, im, y, x, t, d_types, d_d):
	p = d_types[t, 0]

	bits, prev = 0, -1
	U = 0
	
	pix = im[y, x]

	for i in range(0, p*2, 2):
		bits <<= 1
		bits |= pix <= im[y + d_d[t, i], x + d_d[t, i+1]]

		if prev != -1:
			U += (prev ^ bits) & 1
		prev = bits

	U += (bits ^ (bits >> (p-1))) & 1

	if U>2:
		res[y, x] = p+1
	else:
		res[y, x] = bits & ((1<<p) - 1)



@cuda.jit
def lbp_ms_ker(res, im, t, d_types, d_d):
	r = 2
	if t == 0:
		r = 1
	i, j = cuda.grid(2)

	#8_1
	if i >= r and j >= r and i+r < 64 and j+r < 64:
		lbp(res, im, i, j, t, d_types, d_d)
	else:
		res[i, j] = -1



#im ==> imagine color redimensionata(64, 64)

def feature_extractor(im):
	if im.shape != (64, 64, 3):
		im = cv2.resize(im, (64, 64))
	#imagine grayscale
	im_g 	= cv2.equalizeHist(cv2.cvtColor	(im, 	cv2.COLOR_BGR2GRAY))
	

	cuIm_g 	= cp.array		(im_g, 		dtype=cp.uint8)
	cuIm 	= cp.array		(im, 		dtype=cp.uint8)
	
	#rezultate intermediare
	lbp_res = cp.empty((64, 64), dtype=cp.int32)

	#vector trasaturi
	res = cp.array([], dtype=cp.uint32)
	bins8_l = bins8.shape[0]
	
	lbp_ms_ker[(4, 2), (16, 32)](lbp_res, cuIm_g, 0, d_types, d_d)
	for i in range(9):
		res = cp.hstack((res, cp.histogram(lbp_res[corners[i][0]:corners[i][0]+26, corners[i][1]:corners[i][1]+26], bins=bins8)[0]))

	lbp_ms_ker[(4, 2), (16, 32)](lbp_res, cuIm_g, 1, d_types, d_d)
	res = cp.hstack((res, cp.histogram(lbp_res, bins=bins8)[0]))
	

	lbp_ms_ker[(4, 2), (16, 32)](lbp_res, cuIm_g, 2, d_types, d_d)
	res = cp.hstack((res, cp.histogram(lbp_res, bins=bins16)[0]))
	
	
	return cp.asnumpy(res)

@cuda.jit
def histogram_rg_reduce(res, hr, hg):
	i = cuda.grid(1)
	if hr[i] > hg[i]:
		res[i] = hr[i] - hg[i]
	else:
		res[i] = hg[i] - hr[i]


def feature_extractor_extended(im):
	im = cv2.resize(im, (64, 64))
	fv = feature_extractor(im)

	cuIm_r = cp.array(im[:, :, 2], dtype=cp.uint8) #red   ch
	cuIm_g = cp.array(im[:, :, 1], dtype=cp.uint8) #green ch

	res_g = cp.array([], dtype=cp.uint32)
	res_r = cp.array([], dtype=cp.uint32)

	lbp_res = cp.empty((64, 64), dtype=cp.int32)

	lbp_ms_ker[(4, 2), (16, 32)](lbp_res, cuIm_g, 0, d_types, d_d)
	for i in range(9):
		res_g = cp.hstack((res_g, cp.histogram(lbp_res[corners[i][0]:corners[i][0]+26, corners[i][1]:corners[i][1]+26], bins=bins8)[0]))

	lbp_ms_ker[(4, 2), (16, 32)](lbp_res, cuIm_r, 0, d_types, d_d)
	for i in range(9):
		res_r = cp.hstack((res_r, cp.histogram(lbp_res[corners[i][0]:corners[i][0]+26, corners[i][1]:corners[i][1]+26], bins=bins8)[0]))

	histogram_rg_reduce[(1,), (531, )](res_r, res_r, res_g)

	res_r = cp.asnumpy(res_r)
	fv = np.hstack((fv, res_r))
	return fv



@njit
def bins_init(bit_width, res=np.array([], dtype=np.uint32)):
	for i in np.arange(1<<bit_width, dtype=np.uint32):
		temp = i

		bits, U, prev = 0, 0, -1
		for ii in range(bit_width - 1):
			if prev >= 0:
				U += (temp&1) ^ prev
			prev = temp&1
			temp >>= 1

		U += (temp&1 ^ prev) + (temp&1 ^ i&1)
		if U <= 2 or i == bit_width + 1:
			res = np.append(res, i)

	res = np.append(res, 1<<bit_width)
	return res
	

bins8 = bins_init(8)
bins16 = bins_init(16)

bins8  = cp.array(bins8, dtype=cp.uint32)
bins16 = cp.array(bins16, dtype=cp.uint32)
	
@cuda.jit
def add_ker(res, x, y):
	i = cuda.grid(1)
	res[i] = x[i] + y[i]



if __name__ == "__main__":
	cwd = Path("C:\\Users\\vlads\\Desktop")
	impath = Path.joinpath(cwd, "download.jfif")

	im = cv2.imread(str(impath))

	fv = feature_extractor(im)

	print(fv, fv.shape)


 