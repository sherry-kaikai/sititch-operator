#coding: utf-8
import numpy as np
import cv2
import time
# # case 1
# img1 = cv2.imread('../images/simple/1.jpg') 
# img2 = cv2.imread('../images/simple/2.jpg')

# case 2
img1 = cv2.imread('./multi_stitch/Python-Multiple-Image-Stitching-master/images/S1.jpg')
img2 = cv2.imread('./multi_stitch/Python-Multiple-Image-Stitching-master/images/S2.jpg')


start =time.time()

# surf = cv2.SIFT_create()
surf = cv2.ORB_create()
kp1,des1=surf.detectAndCompute(img1,None)  #查找关键点和描述符
kp2,des2=surf.detectAndCompute(img2,None)
 
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
M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC, 5.0)         #生成变换矩阵

# 这是简易的拼接方法
# H = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC, 5.0)    
# h,w=img1.shape[:2]
# h1,w1=img2.shape[:2]
# shft=np.array([[1.0,0,w],[0,1.0,0],[0,0,1.0]])
# M=np.dot(shft,H[0])            #获取左边图像到右边图像的投影映射关系
# dst_corners=cv2.warpPerspective(img1,M,(w*2,h))#透视变换，新图像可容纳完整的两幅图
# # cv2.imwrite('./tiledImg1.png',dst_corners)   #显示，第一幅图已在标准位置
# dst_corners[0:h,w:w*2]=img2  #将第二幅图放在右侧

# end = time.time()
# print('Running time: %s Seconds'%(end-start))

# cv2.imwrite('./tiledImg.png',dst_corners)
# cv2.imwrite('./img1.png',img1)
# cv2.imwrite('./img2.png',img2)


warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))#透视变换
direct = warpImg.copy()

direct[0:img1.shape[0], 0:img1.shape[1]] = img1
cv2.imwrite("./2ori_direct.png", direct) # 直接拼接，有明显缝隙
simple = time.time()
rows, cols = img1.shape[:2]

for col in range(0, cols):
    if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
        left = col
        break
for col in range(cols - 1, 0, -1):
    if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
        right = col
        break

blending = time.time()
res = np.zeros([rows, cols, 3], np.uint8)
for row in range(0, rows):
    for col in range(0, cols):
        if not img1[row, col].any():  # 如果没有原图，用旋转的填充
            res[row, col] = warpImg[row, col]
        elif not warpImg[row, col].any():
            res[row, col] = img1[row, col]
        else: #加权处理，重叠部分，离左边图近的，左边图的权重就高一些，离右边近的，右边旋转图的权重就高一些，然后两者相加，使得过渡是平滑地
            srcImgLen = float(abs(col - left))
            testImgLen = float(abs(col - right))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
final = time.time()

print("simple stich cost %f" % (simple - start))
print("\nblending cost %f" % (final - blending))
print("\ntotal cost %f" % (final - start))

cv2.imwrite("./2warp.png", warpImg) # 加权拼接
