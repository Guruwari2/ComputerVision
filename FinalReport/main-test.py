'''open_image(image1)
open_image(image2)


prewarp(image1)
prewarp(image2)


morphing(image1, image2)

postwarp(imageoutput)'''
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from prewarp_process import prewarp
from morphing_process import morphing
from postwarp_process import postwarp


class ViewMorphing(object):
    def __init__(self):
        pass


    def open_image(self, img1_path, img2_path, img1_name = 'Image1', img2_name = 'Image2'):
        self.img1, self.img2 = cv2.imread(img1_path), cv2.imread(img2_path)
        self.img1_name, self.img2_name = img1_name, img2_name

    def prewarp(self):
        self.img1, self.img2 = prewarp(self.img1, self.img2)
        cv2.imwrite('Prewarp/'+self.img1_name+'_prewarped.jpg', self.img1)
        cv2.imwrite('Prewarp/'+self.img2_name+'_prewarped.jpg', self.img2)


    def morphing(self):
        self.img_final = morphing(self.img1, self.img2)
        cv2.imwrite('Morphed/Pictures_morphed.jpg', self.img_final)

    def postwarp(self):
        self.img_final= postwarp(self.img_final)
        cv2.imwrite('Results/final.jpg', self.img_final)



f __name__ == '__main__':
    import sys

    input_path = 'img/src.jpg'
    output_name = 'results/test.jpg'


    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_name = 'results/'+sys.argv[2]


    hr = HazeRemoval()
    hr.open_image(input_path)
    hr.get_dark_channel()
    hr.get_air_light()
    hr.get_transmission()
    hr.guided_filter()
    hr.recover()
    hr.show(output_name)