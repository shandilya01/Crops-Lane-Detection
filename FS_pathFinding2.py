import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import warnings

warnings.filterwarnings("ignore")

from matplotlib.pyplot import imshow as mpimshow

def canny(image):
	blur = cv2.GaussianBlur(image, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny

def points_from_slopes(image , line_intercept):
	try:
		slope, intercept = line_intercept
	except TypeError:
		slope, intercept = 0.001,0

	y1 = image.shape[0]
	y2 = int(image.shape[0]*0.3)
	x1 = int((y1-intercept)/slope);
	x2 = int((y2-intercept)/slope);

	return np.array([x1,y1,x2,y2]);


def average_slope_intercept(image ,lines):
	
	y1 = lane_image.shape[0]
	y2 = int(lane_image.shape[0]*0.3)
	nar = np.array([
                [[25,y1,26,y2]],
                [[375,y1,374,y2]]
                ])
	try:
		lines = np.append(lines,nar, axis=0)
	except ValueError:
		lines = nar

	
	left_fit=[]
	right_fit=[]
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)
		
		x1 = x1+((x2-x1)*(image.shape[0]-y1))/(y2-y1)
		x2 = x1+((x2-x1)*(int(image.shape[0]*0.3)-y1))/(y2-y1)
		y1 = image.shape[0]
		y2 = int(image.shape[0]*0.3)

		if x1<0.5*image.shape[0]:
			left_fit.append((x1,y1,x2,y2))
		else:
			right_fit.append((x1,y1,x2,y2))

	left_line_avg = np.average(left_fit, axis=0)
	right_line_avg = np.average(right_fit ,axis=0)

	new_lines = np.array([left_line_avg, right_line_avg],dtype = int)

	return new_lines


def display_lines(image ,lines):
	line_image = np.zeros_like(image)
	if lines is not None: 
		for line in lines:
			x1,y1,x2,y2 = line.reshape(4)
			cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
	return line_image


def region_of_interest(image):
	# r=lane_image[:,:,0]
	# g=lane_image[:,:,1]
	# b=lane_image[:,:,2]

	# for _ in range(10):
	# 	g = cv2.GaussianBlur(g, (5,5), 0)
	# 	b = cv2.GaussianBlur(b, (5,5), 0)
	# 	r = cv2.GaussianBlur(r, (5,5), 0) 
	
	# greeny = 2*g-b-r
	# for _ in range(10):
	# 	greeny = cv2.medianBlur(greeny,5);

	# greeny = cv2.threshold(greeny, 250, 255,cv2.THRESH_BINARY)[1]

	# img = greeny
	# kernel =np.array([[0,1,0],
	# 									[0,1,0],
	# 									[0,1,0]], 
	# 								np.uint8)
		
	# img_dilation = cv2.dilate(img, kernel, iterations=5)
	# for _ in range(3):
	# 	img_dilation = cv2.medianBlur(img_dilation,5);
	# h = img_dilation.shape[1]
	# polygons = np.array([[(80,h),(320,h),(320,0),(80,0)]])

	# #img_dilation = canny(img_dilation)
	# cv2.fillPoly(img_dilation, polygons, 0)
	# return img_dilation


	hsv = cv2.cvtColor(image , cv2.COLOR_RGB2HSV)
	lower_green= np.array([0,100,0])
	upper_green= np.array([255,255,255])

	greeny = cv2.inRange(hsv, lower_green, upper_green)
	for _ in range(10):
		greeny = cv2.medianBlur(greeny,5)
	
	return canny(greeny)


####################

cap = cv2.VideoCapture("testvideo3.mp4")
while(cap.isOpened()):
	_,frame = cap.read();
	lane_image = np.copy(frame)
	lane_image = cv2.resize(lane_image,(400,400))

	cropped_image = region_of_interest(lane_image)
	h = cropped_image.shape[1]
	polygons = np.array([[(100,h),(300,h),(300,0),(100,0)]])
	cv2.fillPoly(cropped_image, polygons, 0)

	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 70, np.array([]), minLineLength=90, maxLineGap=20)
	
	avg_lines = average_slope_intercept(lane_image, lines)
	line_image = display_lines(lane_image ,avg_lines)
	combo_image = cv2.addWeighted(lane_image,0.8, line_image,1 ,gamma = 0) 
	
	debug = cv2.bitwise_and(lane_image ,lane_image, mask=cropped_image)
	cv2.imshow("result", combo_image)
	
	k = cv2.waitKey(1) & 0xFF
	if k==27:
		break;