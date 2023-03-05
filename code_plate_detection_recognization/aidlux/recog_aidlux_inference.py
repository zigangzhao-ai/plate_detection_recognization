# aidlux相关
from cvs import *
import aidlite_gpu
from utils import *
import time
import cv2
import os 

source ="/home/code_plate_detection_recognization_1/demo/crop"
recog_model_path = "/home/code_plate_detection_recognization_1/weights/LPRNet_Simplified.tflite"
save_dir = "/home/code_plate_detection_recognization_1/demo/output"
aidlite = aidlite_gpu.aidlite()
# Aidlite模型路径
# 定义输入输出shape
# 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
inShape1 =[1 * 3 * 24 * 94 * 4]
outShape1= [1 * 68 * 18 * 4]
aidlite.ANNModel(recog_model_path,inShape1,outShape1,4,2)
# print('cpu:',aidlite.ANNModel(recog_model_path,inShape1,outShape1,4,0))



for img_name in os.listdir(source):
    print(img_name)
    image_ori = cv2.imread(os.path.join(source, img_name))
   
    image_recog = reg_preprocess(image_ori)
    print(image_recog,image_recog.max(), image_recog.min(),type(image_recog),image_recog.shape)
    # recognization inference
   # aidlite.set_g_index(1)
    aidlite.setInput_Float32(image_recog,24,94)
    aidlite.invoke()
    #取得模型的输出数据
    probs = aidlite.getOutput_Float32(0)
    print(probs)
    print(probs.shape)
    probs = np.reshape(probs, (1, 68, 18))
    print(probs)

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
    

    label = f'names{[str(cls)]}'
    print(label)
    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    # plot_one_box_class(xyxy_, image_ori, label=label, predstr=cls,
    #                     line_thickness=3)

# Save results (image with detections)


# cap.release()
# cv2.destroyAllWindows() 