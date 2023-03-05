
from models.LPRNet import CHARS, LPRNet
import numpy as np

def main():
# load PyTorch model 
    import torch
    lprnet_mod = LPRNet(lpr_max_len=8,phase=False, class_num=len(CHARS), dropout_rate=0)
    lprnet_mod.load_state_dict(torch.load('weights/lprnet_best.pth', map_location="cpu"))
    lprnet_mod.eval()
 
#load onnx model 
    import onnxruntime as ort
    lprnet_onnx = ort.InferenceSession("weights/LPRNet_Simplified.onnx")
 
# pytorch inference 
    dummy_torch = torch.randn(1,3,24,94)
    # lprnet_torch.eval()
    with torch.no_grad():
        torch_res = lprnet_mod(dummy_torch).numpy()
#onnx inference 
    dummy_np = dummy_torch.data.numpy()
    onnx_res = lprnet_onnx.run(["139"],{"input.1":dummy_np})[0];  
 
#diff 
    try:
        np.testing.assert_almost_equal(torch_res, onnx_res, decimal=4)
    except AssertionError:
        print("The torch and onnx results are not equal at decimal=4")
    else:
        print("The torch and onnx results are equal at decimal=4")

if __name__ == "__main__":
    main()
