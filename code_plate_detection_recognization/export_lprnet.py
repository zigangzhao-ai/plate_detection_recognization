




"""Exports a LPRNET *.pt model to ONNX  formats

"""



def convert2onnx():
    import torch
    import onnx
    from onnxsim import simplify
    from models.LPRNet import CHARS, LPRNet

    lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    model = torch.load('weights/lprnet_best.pth', map_location="cpu")
    lprnet.load_state_dict(model)
    lprnet.eval()
    dummy_input = torch.randn(1, 3, 24, 94)
    torch.onnx.export(lprnet, dummy_input, "weights/LPRNET.onnx", verbose=True, opset_version=11)
    print("Successful to convert the pytorch model to onnx. ONNX file: model/LPRNET.onnx")
    #简化Onnx模型 
    model_onnx = onnx.load("weights/LPRNET.onnx")
    model_simp, check = simplify(model_onnx)
    assert check
    onnx.save(model_simp, "weights/LPRNet_Simplified.onnx")
    print("Simplified the onnx model to model/LPRNet_Simplified.onnx")


if __name__ == '__main__':
    convert2onnx()