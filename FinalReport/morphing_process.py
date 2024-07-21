import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# Morphing function
#def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac = 0.5, dissolve_frac = 0.5):
def morphing(im1, im2, im1_pts, im2_pts, tri, warp_frac = 0.5):
	im1 = im1.astype(np.float64) / 255.0
	im2 = im2.astype(np.float64) / 255.0
	im1_pts = np.array(im1_pts)
	im2_pts = np.array(im2_pts)
	intermediate_pts = (1 - warp_frac)* im1_pts+ warp_frac*im2_pts
	morphed_im = np.zeros(im1.shape, dtype=im1.dtype)

	#min_x1, min_y1 = np.min(im1_pts[:, 0]),np.min(im1_pts[:, 1])
	#max_x1, max_y1 = np.max(im1_pts[:, 0]),np.max(im1_pts[:, 1])
#
	#corners1 = np.array([[min_x1,min_y1],[max_x1,min_y1],[min_x1,max_y1],[max_x1,max_y1]], dtype=np.float32).reshape(-1, 1, 2)
#
	#min_x2, min_y2 = np.min(im2_pts[:, 0]),np.min(im2_pts[:, 1])
	#max_x2, max_y2 = np.max(im2_pts[:, 0]),np.max(im2_pts[:, 1])
#
	#corners2 = np.array([[min_x2,min_y2],[max_x2,min_y2],[min_x2,max_y2],[max_x2,max_y2]], dtype=np.float32).reshape(-1, 1, 2)


	for index,t in enumerate(tri.simplices):
		x1, y1, z1 = im1_pts[t].astype(np.float32)
		x2, y2, z2 = im2_pts[t].astype(np.float32)
		x, y, z = intermediate_pts[t].astype(np.float32)     

		M1 = cv2.getAffineTransform(np.array([x1, y1, z1]), np.array([x, y, z]))
		M2 = cv2.getAffineTransform(np.array([x2, y2, z2]), np.array([x, y, z]))


#
		#transformed_corners1 = cv2.perspectiveTransform(corners1, M1).reshape(-1,2)
		#print(transformed_corners1)
		#tmax_x1, tmax_y1 = np.max(transformed_corners1, axis=0)
		#tmin_x1, tmin_y1 = np.min(transformed_corners1, axis=0)
#
		#transformed_corners2 = cv2.perspectiveTransform(corners2, M2).reshape(-1,2)
#
		#tmax_x2, tmax_y2 = np.max(transformed_corners2, axis=0)        
		#tmin_x2, tmin_y2 = np.min(transformed_corners2, axis=0)
		warped_im1 = cv2.warpAffine(im1, M1, (im1.shape[1], im1.shape[0]), flags=cv2.INTER_LINEAR)
		warped_im2 = cv2.warpAffine(im2, M2, (im2.shape[1], im2.shape[0]), flags=cv2.INTER_LINEAR)
		#warped_im1 = cv2.warpAffine(im1, M1, (int(tmax_x1-tmin_x1), int(tmax_y1-tmax_y2)), flags=cv2.INTER_LINEAR)
		#warped_im2 = cv2.warpAffine(im2, M2, (int(tmax_x2-tmin_x2), int(tmax_y2-tmin_y2)), flags=cv2.INTER_LINEAR)



		mask = np.zeros(im1.shape, dtype=np.float32)
		cv2.fillConvexPoly(mask, np.int32(np.array([x, y, z])), (1.0, 1.0, 1.0), 16, 0)
		#morphed_im = morphed_im * (1 - mask) + (1 - dissolve_frac) * warped_im1 * mask + dissolve_frac * warped_im2 * mask
		morphed_im = morphed_im * (1 - mask) + (1 - warp_frac) * warped_im1 * mask + warp_frac * warped_im2 * mask

	return (morphed_im*255).astype(np.uint8)