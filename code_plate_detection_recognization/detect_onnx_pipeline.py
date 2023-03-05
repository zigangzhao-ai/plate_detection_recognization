import argparse
import time
import onnxruntime as ort
import numpy as np
import cv2
import os
import re
# from utils.datasets import letterbox
from utils.plots import *
from models.LPRNet import CHARS

# 188, 58, 250, 72, 372, 148], [130, 39, 107, 62, 157, 46], [54, 18, 72, 29, 105, 30]
anchor = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)

def st_letterbox_resize_image(_image, dst_width, dst_height, border_value=(128, 128, 128),
                            return_scale=False, interpolation=cv2.INTER_LINEAR):
    src_height, src_width = _image.shape[:2]
    x_scale = dst_width / src_width
    y_scale = dst_height / src_height
    if y_scale > x_scale:
        resize_w = dst_width
        resize_h = int(x_scale * src_height)
        left = right = 0
        top = (dst_height - resize_h) // 2
        bottom = dst_height - resize_h - top
        scale = x_scale
    else:
        resize_w = int(y_scale * src_width)
        resize_h = dst_height
        left = (dst_width - resize_w) // 2
        right = dst_width - resize_w - left
        top = bottom = 0
        scale = y_scale
    resized_image = cv2.resize(_image, (int(resize_w), int(resize_h)), interpolation=interpolation)
    dst_image = cv2.copyMakeBorder(resized_image, int(top), int(bottom), int(left), int(right),
                                    cv2.BORDER_CONSTANT, value=border_value)
    if not return_scale:
        return dst_image
    else:
        return dst_image, scale, left, top

# def make_grid(nx=20, ny=20):
#     yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#     return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def make_grid(nx=20, ny=20):
    nx_vec = np.arange(nx)
    ny_vec = np.arange(ny)
    yv ,xv = np.meshgrid(ny_vec, nx_vec)
    grid = np.stack((yv,xv),axis=2)
    grid = grid.reshape(1,1,ny,nx,2)
    return grid

def det_preprocess(image_ori, imgsz):
    # ---preprocess image for detection
    image, scale, left, top = st_letterbox_resize_image(image_ori, imgsz, imgsz, return_scale=True)
    # image = letterbox(image_ori, imgsz)[0]
    print(image.shape)
    image = np.array(image, dtype=np.float)
    image /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(image.shape) == 3:
        image = np.expand_dims(image, 0)
    image = np.array(image, dtype=np.float32).transpose(0, 3, 1, 2)
    return image,  scale, left, top

def cxcywh2xyxy( x, scale, left, top):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2 - left
    y[:, 2] = x[:, 0] + x[:, 2] / 2 - left
    y[:, 1] = x[:, 1] - x[:, 3] / 2 - top
    y[:, 3] = x[:, 1] + x[:, 3] / 2 - top
    y /= scale
    return y

def non_max_suppression( boxes, confs, iou_thresh=0.3):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    order = confs.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def det_poseprocess(outputs, imgsz, scale, left, top ,conf_thresh, iou_thres):
    #
    stride = [imgsz / output.shape[-2] for output in outputs]  # forward

    # anchor_new = []
    # for i, anchor_ in enumerate(anchor):
    #     anchor_i =np.array([anchor_[i:i+2] for i in range(0, len(anchor_), 2)] ).astype(np.float32)
    #     anchor_new.append(anchor_i) #(3, 3, 2)

    anchor_new = np.array(anchor).reshape(len(anchor), 1, -1, 1, 1, 2)

    z = []
    for i, output in enumerate(outputs):
        output = sigmoid(output)
        _, _, width, height, _ = output.shape
        grid = np.array(make_grid(width, height))
        output[..., 0:2] = (output[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # x,y
        output[..., 2:4] = (output[..., 2:4] * 2) ** 2 * anchor_new[i]  # w, h
        z.append(output.reshape(1, -1, 6))
    pred = np.concatenate((z[0], z[1], z[2]), axis=1)

    # nms
    true_conf = pred[..., 4:5] * pred[..., 5:]
    if isinstance(conf_thresh,list):# 是否不同类使用不同的置信度
        mask = true_conf > np.array(conf_thresh)
    else:
        mask = true_conf > conf_thresh
    mask = np.sum(mask,axis=-1) > 0
    ####

    classes = np.argmax(pred[mask][:, 5:], axis=-1)
    # 按照类将每个框的坐标原点进行改变 比如0类的坐标原点还是0 1类的坐标原点变为10000，这样做的目的就是可以一起做nms不用分类进行nms
    c = classes.reshape((-1, 1)) * 10000  # 这个10000可以是max(W,H)
    # 调整为原图下的坐标
    boxes = cxcywh2xyxy(pred[mask][..., 0:4], scale, left, top)
    nms_boxes = boxes + c
    confs = np.amax(pred[mask][:, 5:] * pred[mask][:, 4:5], 1, keepdims=True)
    keep = non_max_suppression(nms_boxes, confs, iou_thresh=0.5)
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    classes = classes.reshape((-1, 1))
    return boxes, confs, classes
    # pred = non_max_suppression(pred, conf_thres, iou_thres)
    # return pred


def reg_preprocess(xyxy_, image_ori, img_size=[94, 24]):
    img_crop = np.array(image_ori[xyxy_[1]:xyxy_[3], xyxy_[0]:xyxy_[2]])
    # recognization prepocess
    # img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR) #?
    height, width, _ = img_crop.shape
    if height != img_size[1] or width !=img_size[0]:
        img_crop = cv2.resize(img_crop, img_size)
    image_clas = img_crop.astype('float32')
    image_clas -= 127.5
    image_clas *= 0.0078125
    image_clas = np.transpose(image_clas, (2, 0, 1))

    return image_clas


def reg_postprocess(prebs, inference_cfg=None):
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]  # 对每张图片 [68, 18]
        preb_label = list()
        for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:  # 记录重复字符
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # 去除重复字符和空白字符'-'
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)  # 得到最终的无重复字符和无空白字符的序列
    
    return preb_labels



def detect_onnx(save_img=True, save_conf=True):
    source, det_weights, clas_weights, save_dir, view_img, save_txt, imgsz = opt.source, opt.detect_weights, opt.classifi_weights, opt.save_dir, opt.view_img, opt.save_txt, opt.img_size
    iou_thres, conf_thres = opt.iou_thres, opt.conf_thres
    if save_img:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    # read input file path
    imgs_path = []
    if os.path.isfile(source):
        with open(source, 'r') as f:
            lines = f.readlines()
        for line in lines:
            imgs_path.append(line.split("\n")[0])
    elif os.path.isdir(source):
        imgs_path = os.listdir(source)
    else:
        raise Exception("the input file is file or dir")


    # load detection and classfication onnx model
    ort_det_session = ort.InferenceSession(det_weights, providers=['CUDAExecutionProvider'])
    ort_class_session = ort.InferenceSession(clas_weights, providers=['CUDAExecutionProvider'])

    for img in imgs_path:
        print(img)
        image_ori = cv2.imread(os.path.join(source, img))
        image,  scale, left, top = det_preprocess(image_ori, imgsz)
        # detection inference

        outputs = []
        for i in range(len(anchor)):
            output = ort_det_session.run([ort_det_session.get_outputs()[i].name], input_feed={'images': image})[0]
            outputs.append(output)

        boxes, confs, classes = det_poseprocess(outputs, imgsz, scale, left, top,conf_thres, iou_thres )
        pred = np.hstack((boxes, confs,classes)).astype(np.float32, copy=False)
        print("box",pred)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_ori.shape).round()

                #for detection in reversed(det):
                xyxy = det[:4]
                if xyxy.min()<0:
                    continue
                
                conf = det[4]
                cls = det[5:]
                # Apply Classifier
                xyxy = np.reshape(xyxy, (1, 4))
                xyxy_ = np.copy(xyxy).tolist()[0]
                xyxy_ = [int(i) for i in xyxy_]
                if (xyxy_[2] -xyxy_[0])/(xyxy_[3]-xyxy_[1])>6 or (xyxy_[2] -xyxy_[0])<100:
                    continue


                image_clas = reg_preprocess(xyxy_, image_ori)
                

                # recognization inference
                probs = ort_class_session.run(output_names=['139'], input_feed={'input.1': [image_clas]})[0]
                # print(probs)

                # proprocess
                probs = reg_postprocess(probs)
                # print("pred_str", probs)
                for prob in probs:
                    lb = ""
                    for i in prob:
                        lb += CHARS[i]
                    cls = lb


                label = f'names{[str(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                plot_one_box_class(xyxy_, image_ori, label=label, predstr=cls,
                                    line_thickness=3)

            # Save results (image with detections)
            if save_img:
                img_path = os.path.join(save_dir, img.split("/")[-1])
                cv2.imwrite(img_path, image_ori)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_weights', nargs='+', type=str,
                        default=r"D:\code_plate_detection_recognization_1\weights\yolov5_best.onnx",
                        help='detection model path(s)')
    parser.add_argument('--classifi_weights', nargs='+', type=str,
                        default=r"D:\code_plate_detection_recognization_1\weights\LPRNet_Simplified.onnx",
                        help='classification model path(s)')
    parser.add_argument('--source', type=str, default=r"D:\code_plate_detection_recognization_1\demo/images",
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_dir', type=str, default=r"D:\code_plate_detection_recognization_1\demo\output",
                        help='source')  # folder,
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, help='display results')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='final_predict', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        startt = time.time()
        detect_onnx()
        print(time.time() - startt)



