import cv2
import numpy as np

def CvtColor(img):
    img_height,img_width,img_channels=img.shape

    HSV_result=np.ones([img_height,img_width,img_channels],dtype='uint8')

    Blue=0
    Green=0
    Red=0

    for h in range(img_height):
        for w in range(img_width):
            for c in range(img_channels):
                if c==0: #依序存放BGR值
                    Blue=img[h][w][c]
                if c==1:
                    Green=img[h][w][c]
                if c==2:
                    Red=img[h][w][c]

                    HSV_result[h][w][0]=H_calculation(Blue,Green,Red)
                    HSV_result[h][w][1]=S_calculation(Blue,Green,Red)
                    HSV_result[h][w][2]=V_calculation(Blue,Green,Red)

    return HSV_result

#B G R S V介於0到1
#H 0到360
def H_calculation(B,G,R):
    B=B/255
    G=G/255
    R=R/255
    Max=max(B,G,R)
    Min=min(B,G,R)
    
    if Max==Min:
        H=0
    elif Max==R and G>=B:
        H=60* (G-B)/(Max-Min)+0
    elif Max==R and G<B:
        H=60* (G-B)/(Max-Min)+360
    elif Max==G:
        H=60* (B-R)/(Max-Min)+120
    elif Max==B:
        H=60* (R-G)/(Max-Min)+240
    return H

def S_calculation(B,G,R):
    B=B/255
    G=G/255
    R=R/255
    Max=max(B,G,R)
    Min=min(B,G,R)
    if Max==0:
        S=0
    else:
        S=1-(Min/Max)
    return S*255 #因為S出來的值介於0~1 但BGR是0~255 所以把除掉的乘回來
def V_calculation(B,G,R):
    B=B/255
    G=G/255
    R=R/255
    Max=max(B,G,R)
    Min=min(B,G,R)
    V=Max
    return V*255#因為V出來的值介於0~1 但BGR是0~255 所以把除掉的乘回來

def dilation(img):
    img_height,img_width=img.shape
    result=np.zeros([img_height,img_width],dtype='uint8')
    for h in range(1,img_height-1):# 這邊先處理邊界以外的部分
        for w in range(1,img_width-1):
            if img[h-1][w]==255 or img[h+1][w]==255 or img[h][w+1]==255 or img[h][w-1]==255:
                result[h][w]=255
    if img[0][1] ==255 or img[1][0]==255: #左上角點
        result[0][0]=255
    if img[0][img_width-1-1]==255 or img[1][img_width-1]==255: #右上角點
        result[0][img_width-1]=255
    if img[img_height-1-1][0]==255 or img[img_height-1][1]==255:#左下角
        result[img_height-1][0]==255
    if img[img_height-1][img_width-1-1]==255 or img[img_height-1-1][img_width-1]==255:#右下角
        result[img_height-1][img_width-1]==255

    for w in range(1,img_width-1-1): #上面的邊
        if img[0][w-1]==255 or img[0][w+1]==255 or img[1][w]==255:
            result[0][w]=255
    for w in range(1,img_width-1-1): #下面的邊
        if img[img_height-1][w-1]==255 or img[img_height-1][w+1]==255 or img[img_height-1-1][w]==255:
            result[img_height-1][w]=255
    for h in range(1,img_height-1-1): #左邊
        if img[h-1][0]==255 or img[h+1][0]==255 or img[h][1]==255:
            result[h][0]=255
    for h in range(1,img_height-1-1): #右邊
        if img[h-1][img_width-1]==255 or img[h+1][img_width-1]==255 or img[h][img_width-1-1]==255:
            result[h][img_width-1]=255
    return result
def erosion(img):
    img_height,img_width=img.shape
    result=np.zeros([img_height,img_width],dtype='uint8')
    for h in range(1,img_height-1):# 這邊先處理邊界以外的部分
        for w in range(1,img_width-1):
            if img[h-1][w]==255 and img[h+1][w]==255 and img[h][w+1]==255 and img[h][w-1]==255:
                result[h][w]=255
    if img[0][1] ==255 and img[1][0]==255: #左上角點
        result[0][0]=255
    if img[0][img_width-1-1]==255 and img[1][img_width-1]==255: #右上角點
        result[0][img_width-1]=255
    if img[img_height-1-1][0]==255 and img[img_height-1][1]==255:#左下角
        result[img_height-1][0]==255
    if img[img_height-1][img_width-1-1]==255 and img[img_height-1-1][img_width-1]==255:#右下角
        result[img_height-1][img_width-1]==255

    for w in range(1,img_width-1-1): #上面的邊
        if img[0][w-1]==255 and img[0][w+1]==255 and img[1][w]==255:
            result[0][w]=255
    for w in range(1,img_width-1-1): #下面的邊
        if img[img_height-1][w-1]==255 and img[img_height-1][w+1]==255 and img[img_height-1-1][w]==255:
            result[img_height-1][w]=255
    for h in range(1,img_height-1-1): #左邊
        if img[h-1][0]==255 and img[h+1][0]==255 and img[h][1]==255:
            result[h][0]=255
    for h in range(1,img_height-1-1): #右邊
        if img[h-1][img_width-1]==255 and img[h+1][img_width-1]==255 and img[h][img_width-1-1]==255:
            result[h][img_width-1]=255
    return result
def MorOpen(img): #先erosion 在dilation
    temp=erosion(img)
    result=dilation(temp)
    return result
def MorClose(img): #先dilation 在erosion
    temp=dilation(img)
    result=erosion(temp)
    return result
def HSV2Bin(img):
    img_height,img_width,img_channels=img.shape
    binary=np.zeros([img_height,img_width],dtype='uint8')
    skin_upper=[40,150,255]
    skin_lower=[0,30,60]
    for h in range(img_height):
        for w in range(img_width):
            for c in range(img_channels): #如果BGR值 在上下限之間 則設為白色
                if img[h][w][c]>skin_lower[c] and img[h][w][c]<skin_upper[c]:
                    binary[h][w]=255
                else:
                    binary[h][w]=0 #其中一個不在範圍內則設黑色 並跳開
                    break
    return binary

def BGR2Bin(img):
    img_height,img_width,img_channels=img.shape
    binary=np.zeros([img_height,img_width],dtype='uint8')
    skin_upper=[215,240,236]
    skin_lower=[82,150,177]
    for h in range(img_height):
        for w in range(img_width):
            for c in range(img_channels): #如果HSV值 在上下限之間 則設為白色
                if img[h][w][c]>skin_lower[c] and img[h][w][c]<skin_upper[c]:
                    binary[h][w]=255
                else:
                    binary[h][w]=0#其中一個不在範圍內則設黑色 並跳開
                    break
    return binary
img=cv2.imread("C:/Users/user/Desktop/imageprocess/HW2/hand_3.jpg")

cv2.imshow("Original",img)
cv2.waitKey()

result_BGR=BGR2Bin(img)
result_BGR=MorOpen(result_BGR)
result_BGR=MorClose(result_BGR)
cv2.imshow("bin_BGR",result_BGR)


self_hsv=CvtColor(img)
result_hsv=HSV2Bin(self_hsv)
result_hsv=MorOpen(result_hsv)
result_hsv=MorClose(result_hsv)
cv2.imshow("bin_HSV",result_hsv)
cv2.waitKey()