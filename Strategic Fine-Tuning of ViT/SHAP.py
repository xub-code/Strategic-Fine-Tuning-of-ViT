# 导入工具包
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
from vit_model import vit_base_patch16_224_in21k as create_model


def shape_ours3(image_path,idx_to_labels_path,checkpoint_path,batch_size,n_evals,outputs,num_classes):
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    model = create_model(num_classes=num_classes, has_logits=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    # model = torch.load(checkpoint_path)
    # model = model.eval().to(device)
    idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()
    print('idx_to_labels:',idx_to_labels)
    class_names = list(idx_to_labels.values())
    print('class_names:',class_names)
    # 载入一张测试图像，整理维度
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize((224, 224))  # 使用PIL的resize方法
    X = torch.Tensor(np.array(img_pil)).unsqueeze(0)
    print('图像维度：',X.shape)

    # 预处理
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x

    def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
        return x

    transform = [
        transforms.Lambda(nhwc_to_nchw),
        transforms.Resize(224),
        transforms.Lambda(lambda x: x * (1 / 255)),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(nchw_to_nhwc),
    ]

    inv_transform = [
        transforms.Lambda(nhwc_to_nchw),
        transforms.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist()
        ),
        transforms.Lambda(nchw_to_nhwc),
    ]

    transform = torchvision.transforms.Compose(transform)
    inv_transform = torchvision.transforms.Compose(inv_transform)

    # 构建模型预测函数
    def predict(img: np.ndarray) -> torch.Tensor:
        img = nhwc_to_nchw(torch.Tensor(img)).to(device)
        output = model(img)
        return output

    # 测试整个工作流正常
    Xtr = transform(X)
    out = predict(Xtr[0:1])
    print('输出：',out.shape)

    classes = torch.argmax(out, axis=1).detach().cpu().numpy()
    print(f'Classes: {classes}: {np.array(class_names)[classes]}')

    # 设置shap可解释性分析算法
    # 构造输入图像
    input_img = Xtr[0].unsqueeze(0)
    print('输入图像形状：',input_img.shape)

    # 定义 mask，遮盖输入图像上的局部区域
    masker_blur = shap.maskers.Image("blur(8, 8)", Xtr[0].shape)
    # 创建可解释分析算法
    explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

    shap_values = explainer(input_img, max_evals=n_evals, batch_size=batch_size, outputs=outputs)

#################################################
    # 提取前3个类别的预测值和 SHAP 值
    sorted_indices = torch.argsort(out, descending=True).cpu().numpy()[0][:topk]
    top_classes = np.array(class_names)[sorted_indices]
    top_values = out[0][sorted_indices].detach().cpu().numpy()

    print("Top 3 Predicted Classes with SHAP Values:")
    for i, cls in enumerate(top_classes):
        print(f"Class: {cls}, Prediction Score: {top_values[i]:.4f}")
    ##################################################



    # 设置字体为Times New Roman，字体大小为12
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 12
    # 整理张量维度
    shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]  # 原图
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]  # shap值热力图
    print('原图形状：',shap_values.data.shape)
    print('shap值热力图:',shap_values.values[0].shape)

    # # 可视化
    # shap.image_plot(shap_values=shap_values.values,
    #                 pixel_values=shap_values.data,
    #                 labels=shap_values.output_names)

    # 可视化 SHAP 图像
    # fig, ax = plt.subplots(figsize=(10, 6))
    shap.image_plot(shap_values=shap_values.values,
                    pixel_values=shap_values.data,
                    labels=shap_values.output_names,
                    show=False)

    # 添加标题显示前3个类别及其预测分数
    title_text = "SHAP Values:" + " ".join(
        [f"{cls}: {score:.4f}" for cls, score in zip(top_classes, top_values)]
    )
    plt.suptitle(title_text, x=0.5, y=0.9, fontsize=14, ha='center')

    plt.show()


if __name__ == '__main__':
    image_path=r"XAI/MCI5.png"
    idx_to_labels_path='idx_to_labels.npy'
    checkpoint_path=r"weights/last.pth"
    batch_size = 64
    n_evals = 3000  # SHAP迭代次数越大，显著性分析粒度越精细，计算消耗时间越长3000
    # 前k个预测类别
    topk = 3
    num_classes=3
    outputs = shap.Explanation.argsort.flip[:topk]
    shape_ours3(image_path,idx_to_labels_path,checkpoint_path,batch_size,n_evals,outputs,num_classes=num_classes)