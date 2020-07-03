import numpy as np
import cv2

sigma_list = [15, 80, 250]
def singleScaleRetinex(img, sigma):

	retinex = np.log(img) - np.log(cv2.GaussianBlur(img, (0, 0), sigma))

	return np.exp(retinex)

def multiScaleRetinex(img):
	retinex = np.zeros_like(img, dtype=np.float32)

	for sigma in sigma_list:
		retinex += singleScaleRetinex(img, sigma)

	retinex = retinex / len(sigma_list)

	return retinex#(retinex * 255).astype(np.uint8)

if __name__ == '__main__':
	im = cv2.imread('/home/uzzi/Downloads/rose_photos/real/100.png')
	ret = multiScaleRetinex(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

	while True:
		cv2.imshow('im', cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
		cv2.imshow('im1', ret)

		c = cv2.waitKey(10)
		if c == 27:
			cv2.destroyAllWindows()
			break