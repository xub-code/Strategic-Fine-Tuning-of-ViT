# 导入工具包
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from captum.attr import Occlusion
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model

def Captum_ours3(image_path,checkpoint_path,idx_to_labels_path,num_classes):
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # 载入自己训练好的模型
    # model = torch.load(checkpoint_path)
    # model = model.eval().to(device)
    model = create_model(num_classes=num_classes, has_logits=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()


    # img_pil = Image.open(image_path)
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize((224, 224))  # 使用PIL的resize方法
    # 载入类别
    idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()


    # 缩放、裁剪、转 Tensor、归一化
    transform_A = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    transform_B = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])

    # 缩放、裁剪
    rc_img = transform_A(img_pil)
    # 调整数据维度
    rc_img_norm = np.transpose(rc_img.squeeze().cpu().detach().numpy(), (1, 2, 0))
    # 归一化
    input_tensor = transform_B(rc_img).unsqueeze(0).to(device)

    # 前向预测
    pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    pred_conf, pred_id = torch.topk(pred_softmax, 1)
    pred_conf = pred_conf.detach().cpu().numpy().squeeze().item()
    pred_id = pred_id.detach().cpu().numpy().squeeze().item()
    pred_label = idx_to_labels[pred_id]
    print('预测类别的ID {} 名称 {} 置信度 {:.2f}'.format(pred_id, pred_label, pred_conf))
    # 遮挡可解释性分析
    # 在输入图像上，用遮挡滑块，滑动遮挡不同区域，探索哪些区域被遮挡后会显著影响模型的分类决策。
    # 提示：因为每次遮挡都需要分别单独预测，因此代码运行可能需要较长时间。
    occlusion = Occlusion(model)
    # 小遮挡滑块（运行时间较长，2分钟左右）
    # 更改遮挡滑块的尺寸
    attributions_occ = occlusion.attribute(input_tensor,
                                           strides=(3, 2, 2),  # 遮挡滑动移动步长
                                           target=pred_id,  # 目标类别
                                           sliding_window_shapes=(3, 4, 4),  # 遮挡滑块尺寸
                                           baselines=0)

    # 转为 224 x 224 x 3的数据维度
    attributions_occ_norm = np.transpose(attributions_occ.detach().cpu().squeeze().numpy(), (1, 2, 0))

    viz.visualize_image_attr_multiple(attributions_occ_norm,  # 224 224 3
                                      rc_img_norm,  # 224 224 3
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2)
    plt.show()



if __name__ == '__main__':
    # r"D:\pythonProject\vision_transformer\ADMCI\test\AD\AD018_S_0633a115.png"
    # r"D:\pythonProject\vision_transformer\ADMCI\test\CN\CN023_S_0061a091.png"
    # r"D:\pythonProject\vision_transformer\ADMCI\test\MCI\CI031_S_0830a085.png"
    image_path=r"XAI/CN5.png"
    idx_to_labels_path='idx_to_labels.npy'
    checkpoint_path=r"weights/last.pth"
    num_classes=3
    Captum_ours3(image_path,checkpoint_path,idx_to_labels_path,num_classes=num_classes)
