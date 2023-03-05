"""
@Author: dadao
@Date: 2022/12/03  10:47
"""

import shutil
import cv2
import os



def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        print(filename)
        if not "-" in filename: #对于np等无标签的图片，过滤
            continue
        subname = filename.split("-", 3)[2]  # 第一次分割，以减号'-'做分割,提取车牌两角坐标
        extension = filename.split(".", 1)[1] #判断车牌是否为图片
        if not extension == 'jpg':
            continue
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1) #左上角坐标
        rx, ry = rb.split("&", 1) # 右下角坐标
        width = int(rx) - int(lx) #车牌宽度
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点

        img = cv2.imread(os.path.join(path , filename))
        if img is None:  # 自动删除失效图片（下载过程有的图片会存在无法读取的情况）
            os.remove(os.path.join(path, filename))
            continue
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)[0] +".txt"
        txtfile = os.path.join(txt_path, txtname)
        # 默认车牌为1类，标签为0
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))

def xml_translate(image_path, txt_path,xml_path):
    from xml.dom.minidom import Document

    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    """
    dic = {'0': "plate",  # 创建字典用来对类型进行转,此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
           }
    files = os.listdir(txt_path)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open( os.path.join(txt_path , name))
        txtList = txtFile.readlines()
        for root, dirs, filename in os.walk(image_path):
            img = cv2.imread(os.path.join(root , filename[i]))
            Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = max(int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth), 0)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = max(int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight),0)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = min(int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth),Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = min(int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight),Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束
        xml_save_path = os.path.join(xml_path, name[0:-4] + ".xml")

        f = open(xml_save_path, 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


if __name__ == '__main__':
    # 修改此处地址
    imaged_dir = "/data2/qw/CCPD2019/images" # 改成你所在图片的位置
    txt_dir = "/data2/qw/CCPD2019/labels"#改成你需要保存的txt的路径
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
   
    txt_translate(imaged_dir, txt_dir)
    # yolo转xml
    Validation = False
    if Validation:
        xml_dir = "/data2/qw/CCPD2019/xml_labels" #改成保存xml的路径
        if not os.path.exists(xml_dir):
           os.mkdir(xml_dir)
        xml_translate(imaged_dir, txt_dir,xml_dir)
