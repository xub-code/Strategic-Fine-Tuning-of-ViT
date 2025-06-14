import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import  transforms
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
from vit_model import vit_base_patch16_224_in21k as create_model
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np

def visualize_lime_explanation2(img_path,idx_to_labels_path,idx):
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 载入测试图片
    img_pil = Image.open(img_path).convert("RGB")

    num_classes=3
    # Create the model
    model = create_model(num_classes=num_classes, has_logits=False).to(device)
    # Load model weights
    model_weight_path = r"weights/last.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # # 载入模型(全部加载)
    # model_weight_path = r"weights/last.pth"
    # model=torch.load(model_weight_path, map_location=device)
    # model=model.eval()

    idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()
    print(len(idx_to_labels))
    print(idx_to_labels)

    # 预处理
    trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
    trans_A = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        trans_norm
    ])
    trans_B = transforms.Compose([
        transforms.ToTensor(),
        trans_norm
    ])
    trans_C = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    # # 图像分类预测
    # input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
    # pred_logits = model(input_tensor)
    # pred_softmax = F.softmax(pred_logits, dim=1)
    # top_n = pred_softmax.topk(3)

    # 定义分类预测函数
    def batch_predict(images):
        batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    test_pred = batch_predict([trans_C(img_pil)])
    test_pred.squeeze().argmax()

    # LIME可解释性分析
    explainer = lime_image.LimeImageExplainer()
    # explain_instance() 方法会基于给定的输入图像和分类模型，生成该图像的局部解释，突出显示模型决策的关键区域
    explanation = explainer.explain_instance(np.array(trans_C(img_pil)),#将图像对象 img_pil 转换为NumPy数组格式。
                                             batch_predict,  # 分类预测函数
                                             top_labels=3,#LIME解释器中的参数，表示返回前3个最有可能的分类标签
                                             hide_color=1,#如果hide_color=0，表示保留原始图像的颜色；如果设置为其他值，则可以将图像颜色隐藏，仅突出显示重要区域。
                                             # 用于局部训练线性模型。这些扰动图像可以帮助LIME分析哪些区域对模型的分类结果起到了关键作用。
                                             num_samples=5000)  # LIME生成的邻域图像个数

    print(explanation.top_labels[0])
    '''
    {
    "0": "AD",
    "1": "CN",
    "2": "MCI"
    }
    '''
    # 设置字体为Times New Roman，字体大小为12
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 12
    temp, mask = explanation.get_image_and_mask(idx,
                                                #这是一个布尔参数，用于指定掩码是否只显示“正向”特征（即对模型预测结果贡献较大的正向特征）。
                                                # 如果设置为 True，掩码只会高亮显示对模型决策有正面贡献的区域；如果设置为 False，则显示所有特征，包括正面和负面贡献的区域。
                                                positive_only=False,
                                                # 这是一个整数，指定要显示的特征数量（例如，图像中模型认为最重要的 20 个特征）。
                                                # num_features=20 表示我们只关心最重要的 20 个特征或区域。
                                                # 在图像解释中，这通常意味着突出显示对模型分类最具影响力的 20 个像素或区域
                                                num_features=20,
                                                # 这是一个布尔参数，用于控制是否隐藏不重要的特征。如果设置为 True，未选中的区域将被隐藏（通常被涂成透明或黑色）；
                                                # 如果设置为 False，则不会隐藏这些区域，它们仍然可见。
                                                hide_rest=False)
    # img_boundry = mark_boundaries(temp / 255.0, mask)
    # plt.imshow(img_boundry)
    # plt.show()

    # 将图像的像素值归一化（0~1）到（0~255），然后转换为uint8类型，OpenCV需要这种格式
    temp = (temp / 255.0 * 255).astype(np.uint8)
    # 使用mark_boundaries绘制边界，返回的是一个RGB图像
    img_boundry = mark_boundaries(temp / 255.0, mask)
    # 将图像从浮点数值（0-1）转换到（0-255），同时确保数据类型为uint8以便OpenCV支持
    img_boundry = (img_boundry * 255).astype(np.uint8)
    # OpenCV处理BGR格式，将RGB格式转换为BGR
    img_boundry_bgr = cv2.cvtColor(img_boundry, cv2.COLOR_RGB2BGR)
    # 显示图像
    cv2.imshow("Explanation Boundary", img_boundry_bgr)
    cv2.waitKey(0)  # 等待键盘输入，按任意键关闭窗口
    cv2.destroyAllWindows()

    # 保存图像
    cv2.imwrite("LIME.png", img_boundry_bgr)

if __name__ == '__main__':
    # 示例调用

    img_path = r"D:\pythonProject\vision_transformer\ADMCI\test\MCI\CI136_S_0195a094.png"
    idx_to_labels_path='idx_to_labels.npy'
    idx=0
    visualize_lime_explanation2(img_path,idx_to_labels_path,idx)


