from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# from ResNet.Test.A import img_rgb
from vision_transformer.vit_model import vit_base_patch16_224_in21k as create_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
checkpoint=r'weights\last.pth'
classes=3
model = create_model(num_classes=classes, has_logits=False).to(device)
print(model)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda() #如果是gpu的话加速


def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 创建 GradCAM 对象
"""
GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, 
XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
"""

cam = GradCAM(model=model,
              # target_layers=[model.blocks[-1].norm1],
              target_layers=[model.blocks[-1].norm1],
              # 这里的target_layer要看模型情况，调试时自己打印下model吧
              # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
              # 或者target_layers = [model.blocks[-1].ffn.norm]
              reshape_transform=reshape_transform)



image_path = r"XAI/CN3.png"

rgb_img_ori = cv2.imread(image_path, 1)[:, :, ::-1]

rgb_img = cv2.resize(rgb_img_ori, (224, 224))

# 预处理图像
input_tensor = preprocess_image(
    rgb_img,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5])

# 看情况将图像转换为批量形式
# input_tensor = input_tensor.unsqueeze(0)
if use_cuda:
    input_tensor = input_tensor.cuda()

# 计算 grad-cam
target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
grayscale_cam = grayscale_cam[0, :]

# 将 grad-cam 的输出叠加到原始图像上
#visualization = show_cam_on_image(rgb_img, grayscale_cam)，借鉴的代码rgb格式不对，换成下面
visualization = show_cam_on_image(rgb_img.astype(dtype=np.float32)/255,grayscale_cam)

h,w=rgb_img_ori.shape[:2]
visualization=cv2.resize(visualization,(w,h),interpolation=cv2.INTER_AREA)

# 保存可视化结果
# cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)##注意自己图像格式，吐过本身就是BGR，就不用这行

# 假设 `visualization` 是您计算得到的 CAM 可视化图像
cv2.imshow('CAM Visualization', visualization)  # 显示图片
# 等待按键事件后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('cam.jpg', visualization)



