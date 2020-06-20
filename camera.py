import cv2
import numpy as np

cap = cv2.VideoCapture(0)
clf_face_det = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

def testDevice(source):
   cap = cv2.VideoCapture(source) 
   if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', source)



while True:
	#im = cv2.imread("C:\\Users\\vlads\\Desktop\\download.jpg")
	print(cap.isOpened())

	ret, im = cap.read()
	if im is None:
		continue

	faces = clf_face_det.detectMultiScale(im, scaleFactor=1.3, minSize=(30, 30))

	if len(faces) > 0:
		(x, y, w, h) = faces[0]
		cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0))

	cv2.imshow('img', im)
	c = cv2.waitKey(30)
	if c==27:
		cv2.destroyAllWindows()
		cap.release()
		break

