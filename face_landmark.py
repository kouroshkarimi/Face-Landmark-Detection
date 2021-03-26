
import cv2
import numpy as np
import imutils
import dlib


def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return x, y, w, h



def shape_to_numpy(shape, dtype = 'int'):

	coords = np.ones((68, 2), dtype = dtype)

	for i in range(68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

IMAGE = 'feynman.jpg'

img = cv2.imread(IMAGE)
img = imutils.resize(img, width=800)


pred = dlib.get_frontal_face_detector()
land = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


rects = pred(img,1)

for (i, rect) in enumerate(rects):

	(x, y, w, h) = rect_to_bb(rect)
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	shape = land(img, rect)
	shape = shape_to_numpy(shape)
	
	for (x,y) in shape:

		cv2.circle(img, (x,y),1 , (255, 0, 0), 2)





cv2.imshow("face detect", img)
cv2.waitKey(0)