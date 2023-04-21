import ultralytics
ultralytics.checks()
import cv2
# import torch
import numpy as np
# import sys
from PIL import Image
from ultralytics import YOLO
model = YOLO('best.pt')

import requests
from io import BytesIO
# from google.colab.patches import cv2_imshow

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  #Define COCO Labels
  if labels == []:
    labels = {0: u'Hole', 1: u'Hole1', 2: u'Hole2',3: u'Hole3', 4: u'Hole4'}
  #Define colors
  if colors == []:
    #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2)]

  #plot each boxes
  for box in boxes:
    #add score in label if score=True
    if score :
      label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])+1]
    #filter every box under conf threshold if conf threshold setted
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        box_label(image, box, label, color)
    else:
      color = colors[int(box[-1])]
      box_label(image, box, label, color)

  #show image
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#   try:
#     import google.colab
#     IN_COLAB = True
#   except:
#     IN_COLAB = False

#   if IN_COLAB:
#     cv2_imshow(image) #if used in Colab
#   else :
#     cv2.imshow(image) #if used in Python

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

import os

# 이미지 파일 확장자
extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 이미지 파일 개수 초기화
image_count = 0

# 폴더 경로 설정
folder_path = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image'

# 폴더 내 파일 목록 얻기
file_list = os.listdir(folder_path)
file_names = []
# 파일 목록 순회
for file_name in file_list:
    # 파일의 확장자 추출
    ext = os.path.splitext(file_name)[-1].lower()
    
    # 파일이 이미지 파일이면 개수 증가
    if ext in extensions:
        image_count += 1
        file_names.append(file_name)

for i in range(image_count):
    file_path = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/' + file_names[i]
    test_img = Image.open(file_path)
    save_img_path1 = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/result/1_' + str(i) + '.jpg'
    test_img.save(save_img_path1)
    img = Image.open(save_img_path1)
    img_resize = img.resize((819, 600))
    # 4096 × 3000 
    save_img_path2 = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/result/2_' + str(i) + '.jpg'
    img_resize.save(save_img_path2)
    image_path = save_img_path2

    image = cv2.imread(image_path)

    #회전 추가 
    rows, cols = image.shape[:2]
    res=[]
    for k in range(9):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 10*k, 1)
        dst = cv2.warpAffine(image, M, (cols, rows))
        results = model.predict(dst)

        print(results[0].boxes.data)

        plot_bboxes(dst, results[0].boxes.data, score=False, conf=0.5)
        # save_img_path3 = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/result/3_' + str(i) + '.jpg'

        save_img_path3 = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/result/4_' + str(k) + '.jpg'

        cv2.imwrite(save_img_path3,dst)
        # cv2.imwrite(save_img_path3,image)
        A = []
        for j in range(len(results[0].boxes.data)):
            # print(results[0].boxes.data[i])
            a = results[0].boxes.data[j]
            if a[5]==1 or a[5]==2:
                A.append([a[0],a[1],a[2],a[3]])
        print(A)

        if len(A)!=2:
            print('detect_error')
        else:
            B = []
            for dis in A:
                B.append([(dis[0]+dis[2])/2, (dis[1]+dis[3])/2])
            # print(B)
            Distance = ((B[0][0]-B[1][0])**2 + (B[0][1]-B[0][1])**2)**1/2
            res.append(Distance)
    print(res)
    if max(res) <= 2.5:
        print('OK')
        save_img_path3 = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/result/3_' + str(i)+'OK' + '.jpg'

    else:
        print('NG')
        save_img_path3 = '/Users/hyucksamkwon/project/streamlit_web_deploy_test/deter_image/result/3_' + str(i)+'NG' + '.jpg'
    
    cv2.imwrite(save_img_path3,dst)