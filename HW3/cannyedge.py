import cv2
import numpy as np
import math

img=cv2.imread("./nightmarket.jpg",0)
cv2.imshow("Original",img)
img = cv2.GaussianBlur(img, (3,3),0) 

img_height,img_width=img.shape

M=np.zeros([img_height,img_width],dtype="int16") 
gradient_x=np.zeros([img_height,img_width],dtype="int16") 
gradient_y=np.zeros([img_height,img_width],dtype="int16") 
sobel_y=np.array([(1,2,1),(0,0,0),(-1,-2,-1)], dtype='int')
sobel_x=np.array([(-1,0,1),(-2,0,2),(-1,0,1)], dtype='int')

#計算M
for h in range(1,img_height-1):
        for w in range(1,img_width-1):
            temp_x=0
            temp_y=0
            for filter_X in range(len(sobel_x)):
                for filter_Y in range(len(sobel_x)):

                    temp_x=temp_x+img[h-1+filter_X][w-1+filter_Y]*sobel_x[filter_X][filter_Y]
                    temp_y=temp_y+img[h-1+filter_X][w-1+filter_Y]*sobel_y[filter_X][filter_Y]

            gradient_x[h][w]=temp_x
            gradient_y[h][w]=temp_y
 
            M[h][w]=((temp_x*temp_x)+(temp_y*temp_y))**(0.5)

#Non-Maximum Supperssion
for h in range(1,img_height-1):
        for w in range(1,img_width-1):

            gradient_direction=math.atan(gradient_y[h][w]/gradient_x[h][w])
            gradient_direction=gradient_direction/math.pi*180
            gradient_direction=gradient_direction+360
            gradient_direction=gradient_direction%360
            #print(gradient_direction)
            if gradient_x[h][w]==0:  #垂直方向
                 
                if M[h][w]>M[h-1][w]:
                    M[h-1][w]=0
                    if M[h][w]>M[h+1][w]:
                        M[h+1][w]=0  #M[h][w] 最大
                    else:
                        M[h][w]=0  #M[h+1][w] 最大
                else:
                    M[h][w]=0
                    if M[h-1][w]>M[h+1][w]:
                        M[h+1][w]=0 #M[h-1][w] 最大
                    else:
                        M[h-1][w]=0 #M[h+1][w] 最大
            elif gradient_y[h][w]==0: #水平方向
                if M[h][w]>M[h][w-1]:
                    M[h][w-1]=0
                    if M[h][w]>M[h][w+1]:
                        M[h][w+1]=0  #M[h][w] 最大
                    else:
                        M[h][w]=0  #M[h][w+1] 最大
                else:
                    M[h][w]=0
                    if M[h][w-1]>M[h][w+1]:
                        M[h][w+1]=0 #M[h][w-1] 最大
                    else:
                        M[h][w-1]=0 #M[h][w+1] 最大
           
            elif (gradient_direction>0 and gradient_direction<90) or (gradient_direction>180 and gradient_direction<270): #方向 右上  左下
                if M[h][w]>M[h-1][w-1]:
                    M[h-1][w-1]=0
                    if M[h][w]>M[h+1][w+1]:
                        M[h+1][w+1]=0  #M[h][w] 最大
                    else:
                        M[h][w]=0  #M[h+1][w+1] 最大
                else:
                    M[h][w]=0
                    if M[h-1][w-1]>M[h+1][w+1]:
                        M[h+1][w+1]=0 #M[h-1][w-1] 最大
                    else:
                        M[h-1][w-1]=0 #M[h+1][w+1] 最大

            
            
            elif (gradient_direction>90 and gradient_direction<180) or (gradient_direction>270 and gradient_direction<360): #方向 左下到右上  
                if M[h][w]>M[h-1][w+1]:
                    M[h-1][w+1]=0
                    if M[h][w]>M[h+1][w-1]:
                        M[h+1][w-1]=0  #M[h][w] 最大
                    else:
                        M[h][w]=0  #M[h+1][w-1] 最大
                else:
                    M[h][w]=0
                    if M[h-1][w+1]>M[h+1][w-1]:
                        M[h+1][w-1]=0 #M[h-1][w+1] 最大
                    else:
                        M[h-1][w+1]=0 #M[h+1][w-1] 最大 

#雙門檻和連通成份連接斷掉的邊界 
high_threshold=150
low_threshold=75
for h in range(1,img_height-1):
        for w in range(1,img_width-1):

            if M[h][w]<low_threshold:
                M[h][w]=0
            #elif M[h][w]>120 and M[h][w]<325:

for h in range(1,img_height-1):
        for w in range(1,img_width-1):

            if M[h][w]>=high_threshold:
                if M[h+1][w]>low_threshold:  #上
                    M[h+1][w]=high_threshold
                if M[h-1][w]>low_threshold: #下
                    M[h-1][w]=high_threshold
                if M[h][w-1]>low_threshold: #左
                    M[h][w-1]=high_threshold
                if M[h][w+1]>low_threshold: #右
                    M[h][w+1]=high_threshold
                if M[h+1][w-1]>low_threshold: #左上
                    M[h+1][w-1]=high_threshold
                if M[h-1][w-1]>low_threshold: #左下
                    M[h-1][w-1]=high_threshold
                if M[h+1][w+1]>low_threshold: #右上
                    M[h+1][w+1]=high_threshold
                if M[h-1][w+1]>low_threshold: #右下
                    M[h-1][w+1]=high_threshold

M=np.uint8(M)

signature=cv2.imread("./me_signature.png",0)
signature_y,signature_x=signature.shape


cannyedge=np.copy(M)
for i in range(signature_y):
    for j in range(signature_x):
        if signature[i][j]==255:#如果簽名檔為黑 則self_blur同位置也是黑的
            cannyedge[i][j]=255
cv2.imshow("Canny Edge Detection",cannyedge)
#M canny edge的圖
#1 rho
#np.pi/180 theta
#1 最小投票數
#np.array([]) 替代字符?
#5 線的最小單位
#10 線跟線之間的距離 

#霍夫轉換
lines = cv2.HoughLinesP(M, 1, np.pi/180,5,np.array([]), 5,3)
print(lines.shape)
print(lines[0].shape)
for i in lines:
    x1=i[0][0]
    y1=i[0][1]
    x2=i[0][2]
    y2=i[0][3]
    print(str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2))
    cv2.line(M, (x1, y1), (x2, y2), (255, 255, 255), 2) #在原圖上畫線

for i in range(signature_y):
    for j in range(signature_x):
        if signature[i][j]==255:#如果簽名檔為黑 則self_blur同位置也是黑的
            M[i][j]=255
cv2.imshow("Hough Transform",M)
cv2.waitKey(0)