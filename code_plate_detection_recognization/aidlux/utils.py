import numpy as np
import cv2

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
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
    return image.astype(np.float32),  scale, left, top

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



def preprocess_img(img, target_shape:tuple=None, div_num=255, means:list=[0.485, 0.456, 0.406], stds:list=[0.229, 0.224, 0.225]):
    '''
    图像预处理:
    target_shape: 目标shape
    div_num: 归一化除数
    means: len(means)==图像通道数，通道均值, None不进行zscore
    stds: len(stds)==图像通道数，通道方差, None不进行zscore
    '''
    img_processed = np.copy(img)
    # resize
    if target_shape:
        # img_processed = cv2.resize(img_processed, target_shape)
        img_processed = letterbox(img_processed, target_shape, stride=None, auto=False)[0]

    img_processed = img_processed.astype(np.float32)
    img_processed = img_processed/div_num

    # z-score
    if means is not None and stds is not None:
        means = np.array(means).reshape(1, 1, -1)
        stds = np.array(stds).reshape(1, 1, -1)
        img_processed = (img_processed-means)/stds

    # unsqueeze
    img_processed = img_processed[None, :]

    return img_processed.astype(np.float32).transpose(0, 3, 1, 2)
    
def det_poseprocess(outputs, imgsz, scale, left, top ,conf_thresh, iou_thresh):
    #print(outputs[0].shape[-2],outputs[1].shape[-2],outputs[2].shape[-2])
    # print([output.shape[-2] for output in outputs])
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
    #print("pred",pred)

    # nms
    true_conf = pred[..., 4:5] * pred[..., 5:]
    #print(true_conf)
    if isinstance(conf_thresh,list):# 是否不同类使用不同的置信度
        mask = true_conf > np.array(conf_thresh)
    else:
        mask = true_conf > conf_thresh
    mask = np.sum(mask,axis=-1) > 0
    #print("mask",mask)
    ####

    classes = np.argmax(pred[mask][:, 5:], axis=-1)
    #print("classes",classes)
    # 按照类将每个框的坐标原点进行改变 比如0类的坐标原点还是0 1类的坐标原点变为10000，这样做的目的就是可以一起做nms不用分类进行nms
    c = classes.reshape((-1, 1)) * 10000  # 这个10000可以是max(W,H)
    # 调整为原图下的坐标
    boxes = cxcywh2xyxy(pred[mask][..., 0:4], scale, left, top)
    #print("box",boxes)
    nms_boxes = boxes + c
    confs = np.amax(pred[mask][:, 5:] * pred[mask][:, 4:5], 1, keepdims=True)
    #print("confs",confs)
    keep = non_max_suppression(nms_boxes, confs, iou_thresh=0.5)
    #print("keep",keep)
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    classes = classes.reshape((-1, 1))
    return boxes, confs, classes
    # pred = non_max_suppression(pred, conf_thres, iou_thres)
    # return pred


def reg_preprocess( image_ori, img_size=[94, 24]): #xyxy_,
    
    # recognization prepocess
    # img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR) #
    height, width, _ = img_crop.shape
    if height != img_size[1] or width !=img_size[0]:
        img_crop = cv2.resize(img_crop, img_size)
    image_clas = img_crop.astype('float32')
    image_clas -= 127.5
    image_clas *= 0.0078125
    image_clas = image_clas[None, :]
    image_clas = np.transpose(image_clas, (0, 3, 1, 2)).astype(np.float32)
    # image_clas = np.transpose(image_clas, (2, 0, 1))


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


def plot_one_box_class(x, img, label=None, predstr=None, color=None,  line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        if predstr:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(predstr, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, predstr, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    return cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

