# cannyedge.py

更改圖片路徑

圖片轉成灰階
```
img=cv2.imread("./nightmarket.jpg",0) #第5行 
```

雙門檻值
```
high_threshold=150  #第101行
low_threshold=75    #第102行
```

霍夫轉換
```
lines = cv2.HoughLinesP(M, 1, np.pi/180,5,np.array([]), 5,3)  #第152行
```
* M -> canny edge的圖    
* 1 ->  rho   
* np.pi/180 ->  theta 
* 1 ->  最小投票數    
* np.array([]) ->  替代字符?  
* 5  -> 線的最小單位  
* 10 ->  線跟線之間的距離
