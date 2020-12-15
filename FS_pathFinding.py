import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from matplotlib.pyplot import imshow as mpimshow

#out = cv2.VideoWriter("output1.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output4.avi',fourcc, 10, (640,480))

cap = cv2.VideoCapture("testvideo3.mp4")
while(cap.isOpened()):
	_,image = cap.read();
	lane_image = np.copy(image)
	lane_image = cv2.resize(lane_image , (320,240))

	hsv = cv2.cvtColor(lane_image , cv2.COLOR_RGB2HSV)
	lower_green= np.array([0,90,0])
	upper_green= np.array([255,255,255])

	greeny = cv2.inRange(hsv, lower_green, upper_green)
	for _ in range(10):
	  greeny = cv2.medianBlur(greeny,5)

	n,m  = greeny.shape[:2]
	poly1 = np.array([[(int(m/6),n),(int(5*m/6),n),(int(5*m/6),0),(int(m/6),0)]])
	cv2.fillPoly(greeny, poly1, 0)

	ct_left = cv2.countNonZero(greeny[:,:int(m/2)])
	ct_right = cv2.countNonZero(greeny[:,int(m/2):])

	norm_diff = (abs(ct_left-ct_right)*6)/(n*m)
	inv_slope = norm_diff

	if(ct_left < ct_right):
	  inv_slope = -inv_slope
  
	x1 = int(m/2)
	y1 = n
	y2 = int(0.5*n)
	x2 = x1+int((y2-y1)*inv_slope)

	steer_img = np.zeros_like(lane_image)
	steer_img = cv2.line(steer_img, (x1,y1), (x2,y2), (255,255,0), 10)
	combo_image = cv2.addWeighted(lane_image,0.8, steer_img,1 ,gamma = 0) 
	
	cv2.imshow("result", combo_image)
	cv2.waitKey(10)
	#out.write(combo_image)
	k = cv2.waitKey(1) & 0xFF

	if k==27:
		break;

cap.release() 
out.release() 
cv2.destroyAllWindows() 