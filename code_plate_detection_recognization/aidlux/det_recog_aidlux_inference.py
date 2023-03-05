# aidlux相关
from cvs import *
import aidlite_gpu
from utils import *
import time
import cv2
import os 

anchor = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
source ="/home/code_plate_detection_recognization_1/demo/images"
det_model_path = "/home/code_plate_detection_recognization_1/weights/yolov5.tflite"
recog_model_path = "/home/code_plate_detection_recognization_1/weights/LPRNet_Simplified.tflite"
save_dir = "/home/code_plate_detection_recognization_1/demo/output"
imgsz =640
# AidLite初始化：调用AidLite进行AI模型的加载与推理，需导入aidlite
aidlite = aidlite_gpu.aidlite()
# Aidlite模型路径
# 定义输入输出shape
# 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
aidlite.set_g_index(0)
in_shape0 = [1 * 3* 640 * 640 * 4]
out_shape0 = [1 * 3*40*40 * 6 * 4, 1 * 3*20*20 * 6 * 4, 1 * 3*80*80 * 6 * 4]
aidlite.ANNModel(det_model_path, in_shape0, out_shape0, 4, 0)
# 识别模型 
aidlite.set_g_index(1)
inShape1 =[1 * 3 * 24 * 94 * 4]
outShape1= [1 * 68 * 18 * 4]
aidlite.ANNModel(recog_model_path,inShape1,outShape1,4,-1)



for img_name in os.listdir(source):
    print(img_name)
    image_ori = cv2.imread(os.path.join(source, img_name))
    # frame = cv2.imread("/home/code_plate_detection_recognization_1/demo/images/003748802682-91_84-220&469_341&511-328&514_224&510_224&471_328&475-10_2_5_22_31_31_27-103-12.jpg")
    # img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
    img,  scale, left, top = det_preprocess(image_ori, imgsz=640)
    # 数据转换：因为setTensor_Fp32()需要的是float32类型的数据，所以送入的input的数据需为float32,大多数的开发者都会忘记将图像的数据类型转换为float32
    aidlite.set_g_index(0)
    aidlite.setInput_Float32(img, 640, 640)
    # 模型推理API
    aidlite.invoke()
    # 读取返回的结果
    outputs = [0, 0, 0]
    for i in range(len(anchor)):
        pred = aidlite.getOutput_Float32(i)
    # 数据维度转换
        if pred.shape[0] == 28800:
           pred = pred.reshape(1, 3,40,40, 6)
           outputs[1] = pred           
        if pred.shape[0] == 7200:
           pred = pred.reshape(1, 3,20,20, 6)
           outputs[0] = pred
        if pred.shape[0]== 115200:
           pred = pred.reshape(1, 3,80,80, 6)
           outputs[2] = pred
    # 模型推理后处理
    boxes, confs, classes = det_poseprocess(outputs, imgsz, scale, left, top,conf_thresh=0.3, iou_thresh =0.5)   
    pred = np.hstack((boxes, confs,classes)).astype(np.float32, copy=False)

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            xyxy,conf, cls= det[:4],det[4],det[5:]
            if xyxy.min()<0:
                continue           
            # filter 
            xyxy = np.reshape(xyxy, (1, 4))
            xyxy_ = np.copy(xyxy).tolist()[0]
            xyxy_ = [int(i) for i in xyxy_]
            if (xyxy_[2] -xyxy_[0])/(xyxy_[3]-xyxy_[1])>6 or (xyxy_[2] -xyxy_[0])<100:
                continue
            # image_crop = np.array(image_ori[xyxy_[1]:xyxy_[3], xyxy_[0]:xyxy_[2]])
            # image_crop = np.asarray(image_crop)
            img_crop = np.array(image_ori[xyxy_[1]:xyxy_[3], xyxy_[0]:xyxy_[2]])
            image_recog = reg_preprocess(img_crop)
            print(image_recog.max(), image_recog.min(),type(image_recog),image_recog.shape)
            # recognization inference
            aidlite.set_g_index(1)
            aidlite.setInput_Float32(image_recog,94,24)
            aidlite.invoke()
            #取得模型的输出数据
            probs = aidlite.getOutput_Float32(0)
            print(probs.shape)
            probs = np.reshape(probs, (1, 68, 18))

            print("------",probs)
            # proprocess
            probs = reg_postprocess(probs)
            # print("pred_str", probs)
            for prob in probs:
                lb = ""
                for i in prob:
                    lb += CHARS[i]
                cls = lb

            # result show 
            

            label = f'names{[str(cls)]} {conf:.2f}'
            print(label)
            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            plot_one_box_class(xyxy_, image_ori, label=label, predstr=cls,
                                line_thickness=3)

        # Save results (image with detections)
            img_path = os.path.join(save_dir, img_name)
            # cv2.imwrite(img_path, image_ori)
            cvs.imshow(image_ori)
