import cv2
import numpy as np
import skimage
from skimage.viewer import ImageViewer
import sys
import math

img=cv2.imread("C:/Users/user/Desktop/imageprocess/HW1/me.jpg")

cv2.imshow("Original",img)

print(img.shape)

filter_size=53 #這邊可以改size
extend_size=int((filter_size-1)/2)# 算要額外增加一圈的矩陣寬度是多少
# filter size= 3-> 1   增加一圈
# filter size= 5-> 2   增加二圈  以此類推

img_len,img_width,img_channels=img.shape

y=np.zeros([img_len+(extend_size*2),img_width+(extend_size*2),img_channels],dtype="uint8") #增加兩圈  陣列上下左右增加2倍

for channel in range(img_channels):
    for i in range(extend_size,img_len+extend_size):
        for j in range(extend_size,img_width+extend_size): # 把圖片放進四周圍pixel值為0中
            y[i][j][channel]=img[i-extend_size][j-extend_size][channel]


#卷積運算
def average_blur(img,filter_size):
    img_len,img_width,img_channels=img.shape
    convolution=np.ones([filter_size,filter_size,img_channels])
    for channel in range(img_channels):
        for i in range(img_len):
            for j in range(img_width):
                temp=0
                for filter_X in range(filter_size):
                    for filter_Y in range(filter_size):
                        temp=temp+y[i+filter_X][j+filter_Y][channel]*convolution[filter_X][filter_Y][channel]
                img[i][j][channel]=round(temp/(filter_size*filter_size))

    signature=cv2.imread("C:/Users/user/Desktop/imageprocess/HW1/signature_me.png")
    signature_x,signature_y,signature_channels=signature.shape
    for channel in range(signature_channels):
        for i in range(signature_x):
            for j in range(signature_y):
                if signature[i][j][channel]==0:#如果簽名檔為黑 則self_blur同位置也是黑的
                    img[i][j][channel]=0
    cv2.imshow("self_average_blur",img)
    cv2.waitKey()

#找中位數
def median_blur(img,filter_size):
    img_len,img_width,img_channels=img.shape
    for channel in range(img_channels):
        for i in range(img_len):
            for j in range(img_width):
                temp=[]
                for filter_X in range(filter_size):
                    for filter_Y in range(filter_size):
                        temp.append(y[i+filter_X][j+filter_Y][channel])
                temp.sort() #把filter內的數字排列之後
                half=int(len(temp)/2)#找index一半的元素
                img[i][j][channel]=temp[half]

    signature=cv2.imread("C:/Users/user/Desktop/imageprocess/HW1/signature_me.png")
    signature_x,signature_y,signature_channels=signature.shape
    for channel in range(signature_channels):
        for i in range(signature_x):
            for j in range(signature_y):
                if signature[i][j][channel]==0:#如果簽名檔為黑 則self_blur同位置也是黑的
                    img[i][j][channel]=0
    cv2.imshow("self_median_blur",img)
    cv2.waitKey()

def gaussian_blur(img,filter_size):

    blur = cv2.GaussianBlur(img, (filter_size, filter_size),0)
    signature=cv2.imread("C:/Users/user/Desktop/imageprocess/HW1/signature_me.png")
    signature_x,signature_y,signature_channels=signature.shape
    for channel in range(signature_channels):
        for i in range(signature_x):
            for j in range(signature_y):
                if signature[i][j][channel]==0:#如果簽名檔為黑 則self_blur同位置也是黑的
                    img[i][j][channel]=0
    cv2.imshow("gaussian_blur",img)
    cv2.waitKey()



average_blur(img,filter_size)
median_blur(img,filter_size)
gaussian_blur(img,filter_size)