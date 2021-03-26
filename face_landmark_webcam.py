
# import package
import cv2
import numpy as np
import imutils
import dlib


def rect_to_bb(rect): # convert rect to scaler

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return x, y, w, h


def shape_to_numpy(shape, dtype='int'): # convert shape type (dlib) to numpy

	coords = np.ones((68,2), dtype = dtype)

	for i in range(68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


cam = cv2.VideoCapture(0) # init webcam

predictor = dlib.get_frontal_face_detector() # init face detector
land = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # intit face landmark



while True:

	_ , img = cam.read()
	img = imutils.resize(img, width = 800)


	rects = predictor(img, 1) # predict faces rectangles

	for (i,rect) in enumerate(rects):

		# draw face rectangles
		x, y, w, h = rect_to_bb(rect)
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1) # draw rectangles

		shape = land(img, rect) # compute face landmark
		shape = shape_to_numpy(shape)
	
		for x,y in shape: # draw landmarks (keypoints)
			cv2.circle(img, (x, y), 1, (255, 0, 0), 1)


	cv2.imshow('face landmark', img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break








