import  cupy  	as cp
import  numpy 	as np
import  cv2
import  os
import warnings

from    pathlib import Path
from 	skimage import feature
from    numba import cuda, njit, prange, jit, NumbaDeprecationWarning, NumbaWarning, uint8



warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

d_types = cp.array([[8, 1], [8, 2], [16, 2]], dtype=cp.uint16)

bins8 	= np.array([], dtype=np.uint32)
bins16	= np.array([], dtype=np.uint32)


d_d = cp.array([ [ 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
			   [ 0, -2, 1, -1, 2, 0, 1, 1, 0, 2, -1, 1, -2, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
			   [ 0, -2, 1, -2, 1, -1, 2, -1, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2, -1, 2, -1, 1, -2, 1, -2, 0, -2, -1, -1, -1, -2, -1]
			], dtype=cp.int8)

res_g 		= cp.empty((531,), dtype=cp.uint32)
res_r 		= cp.empty((531,), dtype=cp.uint32)

lbp_res 	= cp.empty((64, 64), dtype=cp.int32)
res 		= cp.empty((1364, ), dtype=cp.uint32)

cuIm		= cp.empty((68, 68, 3), dtype=cp.uint8)
cuIm_g		= cp.empty((68, 68), dtype=cp.uint8)

cuIm[:,  :, :] 	= 0
cuIm_g	[:, :] 	= 0

corners = [ (0, 0), (0, 21), (0, 38),
			(17, 0), (17, 21), (17, 38),
			(38, 0), (38, 21), (38, 38)]
#corners = [(x+2, y+2) for x, y in corners]




@cuda.jit(device=True)
def lbp(res, im, y, x, t, d_types, d_d):
	p = d_types[t, 0]


	bits, prev = 0, -1
	U = 0

	y_im = (y & 31) + 2
	x_im = (x & 31) + 2

	pix = im[y_im, x_im]

	for i in range(0, p*2, 2):
		bits <<= 1
		bits |= pix <= im[y_im + d_d[t, i], x_im + d_d[t, i+1]]

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
	im_sh = cuda.shared.array((36, 36), dtype=uint8)

	i, j = cuda.grid(2)

	i_sh = i&31
	j_sh = j&31

	im_sh[i_sh, j_sh] = im[i, j]
	im_sh[i_sh, j_sh + 4] = im[i, j + 4]
	im_sh[i_sh + 4, j_sh] = im[i + 4, j]
	im_sh[i_sh + 4, j_sh + 4] = im[i + 4, j + 4]

	cuda.syncthreads()
	#im_sh[i + 2, j + 2] = im[i, j]
	#cuda.syncthreads()

	lbp(res, im_sh, i, j, t, d_types, d_d)




#im ==> imagine color redimensionata(64, 64)

@jit(parallel=True)
def feature_extractor(im):
	if im.shape != (64, 64, 3):
		im = cv2.resize(im, (64, 64))

	cuIm_g[2:66, 2:66] 	= cp.array		(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 		dtype=cp.uint8)

	lbp_ms_ker[(2, 2), (32, 32)](lbp_res, cuIm_g, 0, d_types, d_d)

	for i in prange(9):
		res[59*i:59*(i+1)] = cp.histogram(lbp_res[corners[i][0]:corners[i][0]+26, corners[i][1]:corners[i][1]+26], bins=bins8)[0]


	lbp_ms_ker[(2, 2), (32, 32)](lbp_res, cuIm_g, 1, d_types, d_d)
	res[59*9:590] = cp.histogram(lbp_res, bins=bins8)[0]

	lbp_ms_ker[(2, 2), (32, 32)](lbp_res, cuIm_g, 2, d_types, d_d)
	res[590:833] = cp.histogram(lbp_res, bins=bins16)[0]



	return cp.asnumpy(res)

@cuda.jit
def histogram_rg_reduce(res, hr, hg):
	i = cuda.grid(1)
	if hr[i] > hg[i]:
		res[i] = hr[i] - hg[i]
	else:
		res[i] = hg[i] - hr[i]

@jit(parallel=True)
def feature_extractor_extended(im):
	if im.shape != (64, 64, 3):
		im = cv2.resize(im, (64, 64))
	cuIm[2:66, 2:66] = cp.array(im)

	feature_extractor(im)


	cuIm_r = cuIm[:, :, 2]
	cuIm_g = cuIm[:, :, 1]


	lbp_ms_ker[(2, 2), (32, 32)](lbp_res, cuIm_g, 0, d_types, d_d)
	for i in prange(9):
		res_g[59*i:59*(i+1)] = cp.histogram(lbp_res[corners[i][0]:corners[i][0]+26, corners[i][1]:corners[i][1]+26], bins=bins8)[0]

	lbp_ms_ker[(2, 2), (32, 32)](lbp_res, cuIm_r, 0, d_types, d_d)
	for i in prange(9):
		res_r[59*i:59*(i+1)] = cp.histogram(lbp_res[corners[i][0]:corners[i][0]+26, corners[i][1]:corners[i][1]+26], bins=bins8)[0]

	histogram_rg_reduce[(1,), (531, )](res_r, res_r, res_g)


	res[833:1364] = res_r
	return cp.asnumpy(res)



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


