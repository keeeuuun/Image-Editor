import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 
import random

def addNoise(name_img, density):
	img = Image.open(name_img)
	pix = img.load()
	img_width = img.size[0]
	img_height = img.size[1]	

	temp = 0
	while temp <= density:
		x = random.randint(0, img_width - 1)
		y = random.randint(0, img_height - 1)

		R = random.randint(0, 255)
		G = random.randint(0, 255)
		B = random.randint(0, 255)

		pix[x, y] = (R, G, B)

		temp += 1
	img.show()
	img.save('butiful_with_noise.jpg')	

def Lab5(name_img, degreeOfAngle):

	img = cv2.imread(name_img)

	#Гауссовский фильтр
	img_gauss = cv2.GaussianBlur(img,(5,5),0)

	#Фильтр Превитт
	kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img_gauss, -1, kernelx)
	img_prewitty = cv2.filter2D(img_gauss, -1, kernely)
	img_prewitt = img_prewittx + img_prewitty
	
	#Поворот на произвольный угол
	rows, cols, ch = img_prewitt.shape
	M = cv2.getRotationMatrix2D((cols/2, rows/2), degreeOfAngle, 1)
	dst = cv2.warpAffine(img_prewitt, M, (cols, rows))

	cv2.imshow('Gauss', img_gauss)
	#cv2.imshow("Prewitt X", img_prewittx)
	#cv2.imshow("Prewitt Y", img_prewitty)
	cv2.imshow("Prewitt", img_prewittx + img_prewitty)
	cv2.imshow("DST", dst)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	addNoise('butiful.jpg', 5000)
	Lab5('butiful_with_noise.jpg', 80)


main()

			
