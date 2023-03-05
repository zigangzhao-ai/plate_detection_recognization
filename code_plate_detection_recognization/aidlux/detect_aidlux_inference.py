# aidlux相关
from cvs import *
import aidlite_gpu
from utils import *
import time
import cv2
import os 

anchor = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
source ="/home/code_plate_detection_recognization_1/demo/images"
imgsz =640
# AidLite初始化：调用AidLite进行AI模型的加载与推理，需导入aidlite
aidlite = aidlite_gpu.aidlite()
# Aidlite模型路径
model_path = "/home/code_plate_detection_recognization_1/weights/yolov5.tflite"
# 定义输入输出shape
in_shape = [1 * 3* 640 * 640 * 4]
out_shape = [1 * 3*40*40 * 6 * 4,1 * 3*20*20 * 6 * 4,1 * 3*80*80 * 6 * 4]
# 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)

# 识别模型 

for img_name in os.listdir(source):
    print(img_name)
    image_ori = cv2.imread(os.path.join(source, img_name))
    # frame = cv2.imread("/home/code_plate_detection_recognization_1/demo/images/003748802682-91_84-220&469_341&511-328&514_224&510_224&471_328&475-10_2_5_22_31_31_27-103-12.jpg")
    # img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
    img,  scale, left, top = det_preprocess(image_ori, imgsz=640)
    # 数据转换：因为setTensor_Fp32()需要的是float32类型的数据，所以送入的input的数据需为float32,大多数的开发者都会忘记将图像的数据类型转换为float32
    # print(img,img.max(), img.min(),type(img),img.shape)
    aidlite.setInput_Float32(img, 640, 640)
    # 模型推理API
    aidlite.invoke()
    # 读取返回的结果
    outputs = [0,0,0]
    for i in range(len(anchor)):
        pred = aidlite.getOutput_Float32(i)
        # print(pred)
    # 数据维度转换
        if pred.shape[0] ==28800:
           pred = pred.reshape(1, 3,40,40, 6)
           outputs[1] = pred
           
        if pred.shape[0] ==7200:
           pred = pred.reshape(1, 3,20,20, 6)
           outputs[0] = pred
        if pred.shape[0]==115200:
           pred = pred.reshape(1,3,80,80, 6)
           outputs[2] = pred
    #pred = pred.reshape(1, 25200, 6)[0]
    # 模型推理后处理
    boxes, confs, classes = det_poseprocess(outputs, imgsz, scale, left, top,conf_thresh=0.5, iou_thresh =0.3)
    
    pred = np.hstack((boxes, confs,classes)).astype(np.float32, copy=False)
    print("pred",pred)
    # pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.5, iou_thres=0.45)
    # 绘制推理结果
    # res_img = draw_detect_res(image_ori, pred)
    # cvs.imshow(res_img)


# cap.release()
# cv2.destroyAllWindows()  