import numpy as np
import cv2
from math import sqrt

import math

#img 原圖 
#time 縮放倍數
def interpolation(img,time):
    height,width,channels =img.shape
    new_height=int(height*time)
    new_width=int(width*time)

    emptyImage=np.zeros((new_height,new_width,channels),dtype=np.uint8)
    #    -------
    #   | A   B |
    #   | C   D |
    #    -------
    #
    #    -----------
    #   | A       B |
    #   |           |
    #   |           |
    #   | C       D |
    #    -----------
    for channel in range(channels):
        for y in range(new_height):
            for x in range(new_width):
                
                src_X=x/time
                src_Y=y/time
                if src_X%1==0 and src_Y%1==0: #如果XY是整數 就直接取元圖對應到的index
                    src_X=int(src_X)
                    src_Y=int(src_Y)
                    emptyImage[y][x][channel]=img[src_Y][src_X][channel]

                else:
                    A_x=int(src_X)
                    A_y=int(src_Y)

                    B_x=A_x+1
                    B_y=A_y

                    C_x=A_x
                    C_y=A_y+1

                    D_x=A_x+1
                    D_y=A_y+1
                    
                    if D_y>=height:  #D超過的時候 微調ABCD座標
                        D_y=height-1
                        C_y=height-1
                        A_y=C_y-1
                        B_y=D_y-1
                    if D_x>=width:
                        D_x=width-1
                        B_x=width-1
                        A_x=B_x-1
                        C_x=D_x-1

                    A=img[A_y][A_x][channel]
                    B=img[B_y][B_x][channel]
                    C=img[C_y][C_x][channel]
                    D=img[D_y][D_x][channel]

                    A=int(A)
                    B=int(B)
                    C=int(C)
                    D=int(D)


                    w=x-(A_x*time)
                    W=(B_x-A_x)*time
                    h=y-(A_y*time)
                    H=(C_y-A_y)*time

                    i=A+(w*(B-A)/W)
                    j=C+((w*(D-C))/W)
                    
                    pixel=i+((h*(j-i))/H)
                    emptyImage[y][x][channel]=pixel


    return emptyImage
    

def interpolation_test(img,time):
    height,width,channels =img.shape
    new_height=int(height*time)
    new_width=int(width*time)

    emptyImage=np.zeros((new_height,new_width,channels),dtype=np.uint8)
    #    -------
    #   | A   B |
    #   | C   D |
    #    -------
    #
    #    -----------
    #   | A       B |
    #   |           |
    #   |           |
    #   | C       D |
    #    -----------
    for channel in range(channels):
        for y in range(new_height):
            for x in range(new_width):
                
                src_X=x/time
                src_Y=y/time
                if src_X%1==0 and src_Y%1==0: #如果XY是整數 就直接取元圖對應到的index
                    src_X=int(src_X)
                    src_Y=int(src_Y)
                    emptyImage[y][x][channel]=img[src_Y][src_X][channel]

                else:
                    A_x=int(src_X)
                    A_y=int(src_Y)

                    B_x=A_x+1
                    B_y=A_y

                    C_x=A_x
                    C_y=A_y+1

                    D_x=A_x+1
                    D_y=A_y+1
                    
                    if D_y>=height:  #D超過的時候 微調ABCD座標
                        D_y=height-1
                        C_y=height-1
                        A_y=C_y-1
                        B_y=D_y-1
                    if D_x>=width:
                        D_x=width-1
                        B_x=width-1
                        A_x=B_x-1
                        C_x=D_x-1

                    A=img[A_y][A_x][channel]
                    B=img[B_y][B_x][channel]
                    C=img[C_y][C_x][channel]
                    D=img[D_y][D_x][channel]

                    A=int(A)
                    B=int(B)
                    C=int(C)
                    D=int(D)


                    w=x-(A_x*time)
                    W=(B_x-A_x)*time
                    h=y-(A_y*time)
                    H=(C_y-A_y)*time
                    #先從y方向算 再算x
                    i=A+(h*(C-A)/H)
                    j=B+((h*(D-B))/H)
                    
                    pixel=i+((w*(j-i))/W)
                    emptyImage[y][x][channel]=pixel


    return emptyImage
def nearest(img,time):
    height,width,channels =img.shape

    new_height=int(height*time)
    new_width=int(width*time)
    #    -------
    #   | A   B |
    #   | C   D |
    #    -------                
    #    -----------
    #   | A       B |
    #   |           |
    #   |           |
    #   | C       D |
    #    -----------
    emptyImage=np.zeros((new_height,new_width,channels),dtype=np.uint8)
    for channel in range(channels):
        for i in range(new_height):
            for j in range(new_width):
                y=int(i/time) #除放大或縮小的倍率
                x=int(j/time) #取整數
                emptyImage[i][j][channel]=img[y][x][channel]

    return emptyImage
 
img=cv2.imread("C:/Users/user/Desktop/imageprocess/HW1/me.jpg")

print(img.shape)
self_interpolation=interpolation(img,2)

self_nearest=nearest(img,2)
self_interpolation_test=interpolation_test(img,2)

signature=cv2.imread("C:/Users/user/Desktop/imageprocess/HW1/signature_me.png")



signature=interpolation(signature,2)
signature_y,signature_x,channels=signature.shape
for channel in range(channels):
    for y in range(signature_y):
        for x in range(signature_x):
            if signature[y][x][channel]==0:#如果簽名檔為黑 則nearest_neighbor同位置也是黑的
                self_interpolation[y][x][channel]=0
                self_interpolation_test[y][x][channel]=0
                self_nearest[y][x][channel]=0

self_interpolation_height,self_interpolation_width,channels=self_interpolation.shape
for channel in range(channels):
    for y in range(self_interpolation_height):
        for x in range(self_interpolation_width):
            if self_interpolation[y][x][channel]!=self_interpolation_test[y][x][channel]:
                print(self_interpolation[y][x][channel])
                print(self_interpolation_test[y][x][channel])

    print("OK")
cv2.imshow("self_interpolation",self_interpolation)
#cv2.imshow("self_interpolation_test",self_interpolation_test)





cv2.imshow("self_nearest",self_nearest)
cv2.imshow("image",img)
cv2.waitKey(0)

