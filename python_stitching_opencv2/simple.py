#coding: utf-8
import numpy as np
import cv2
import time
# case 1
# leftgray = cv2.imread('../images/simple/S1.jpg')
# rightgray = cv2.imread('../images/simple/S2.jpg')

# case 2
leftgray = cv2.imread('./multi_stitch/Python-Multiple-Image-Stitching-master/images/S1.jpg')
rightgray = cv2.imread('./multi_stitch/Python-Multiple-Image-Stitching-master/images/S2.jpg')



start =time.time()

hessian=400
surf = cv2.xfeatures2d.SURF_create(hessian)
kp1,des1=surf.detectAndCompute(leftgray,None)  #查找关键点和描述符
kp2,des2=surf.detectAndCompute(rightgray,None)
 
FLANN_INDEX_KDTREE=0   #建立FLANN匹配器的参数
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=10) #配置索引，密度树的数量为5
searchParams=dict(checks=50)    #指定递归次数
#FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
flann=cv2.FlannBasedMatcher(indexParams,searchParams)  #建立匹配器
matches=flann.knnMatch(des1,des2,k=2)  #得出匹配的关键点
 
good=[]
#提取优秀的特征点
for m,n in matches:
    if m.distance < 0.7*n.distance: #如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        good.append(m)
 
src_pts = np.array([ kp1[m.queryIdx].pt for m in good])    #查询图像的特征描述子索引
dst_pts = np.array([ kp2[m.trainIdx].pt for m in good])    #训练(模板)图像的特征描述子索引
H=cv2.findHomography(src_pts,dst_pts)         #生成变换矩阵
 
h,w=leftgray.shape[:2]
h1,w1=rightgray.shape[:2]
shft=np.array([[1.0,0,w],[0,1.0,0],[0,0,1.0]])
M=np.dot(shft,H[0])            #获取左边图像到右边图像的投影映射关系
dst_corners=cv2.warpPerspective(leftgray,M,(w*2,h))#透视变换，新图像可容纳完整的两幅图
# cv2.imwrite('./tiledImg1.png',dst_corners)   #显示，第一幅图已在标准位置
dst_corners[0:h,w:w*2]=rightgray  #将第二幅图放在右侧

end = time.time()
print('Running time: %s Seconds'%(end-start))

cv2.imwrite('./tiledImg.png',dst_corners)
cv2.imwrite('./leftgray.png',leftgray)
cv2.imwrite('./rightgray.png',rightgray)