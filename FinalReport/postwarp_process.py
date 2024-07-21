import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matching_process import add_manual_points
#mask_face = cv2.imread('mask/mask_face.jpg')
#mask_face = cv2.imread('mask/bus_mask.jpg')
#mask_face = cv2.resize(mask_face, [image.shape[1], image.shape[0]])


def postwarp_proc(image):
	#final_plane = cv2.imread('mask/mask_face.jpg')
	final_plane = cv2.imread('mask/bus_mask.jpg')
	final_plane = cv2.resize(final_plane, [image.shape[1], image.shape[0]])
	points1, points2 = add_manual_points(image, final_plane)
	H_s, _ = cv2.findHomography(points1, points2)

	postwarp_img = cv2.warpPerspective(image, H_s, (image.shape[0], image.shape[1]))
	return postwarp_img

