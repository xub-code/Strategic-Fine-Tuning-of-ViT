import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from vit_model import vit_base_patch16_224_in21k as create_model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5],
                                                              [0.5, 0.5, 0.5])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    image_path = os.path.join(data_root,  "ADMCI")  # dataset path
    assert os.path.exists(image_path), f"Data path {image_path} does not exist."

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform)

    batch_size = 64
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    net = create_model(num_classes=3, has_logits=False).to(device)
    # Load pre-trained weights
    model_weight_path = "weights/last.pth"
    assert os.path.exists(model_weight_path), f"Cannot find {model_weight_path} file."
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)


    # Read class indices
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), f"Cannot find {json_label_path} file."
    with open(json_label_path, 'r') as json_file:
        class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]

    net.eval()

    y_true = []  # Ground truth labels
    y_scores = []  # Model predicted probabilities

    # Collect all true labels and predicted probabilities
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            outputs = net(val_images)

            # Get the probabilities (softmax)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            y_true.extend(val_labels.numpy())  # Collect true labels
            y_scores.extend(probs)  # Collect predicted probabilities

    # Binarize the true labels for multi-class (One-hot encoding)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    # 设定类别名称
    class_names = ['AD', 'CN', 'MCI']

    # Plot PR curve for each class
    plt.figure(figsize=(8, 6))

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'

    # 设置图形背景色
    plt.gcf().set_facecolor('whitesmoke')  # 设置整个图形背景色

    ax = plt.gca()  # 获取当前坐标轴
    ax.set_facecolor((230 / 255, 230 / 255, 230 / 255))  # 设置坐标轴背景色为相同的浅灰色

    for i in range(3):  # Assuming there are 3 classes
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], np.array([score[i] for score in y_scores]))
        ap = average_precision_score(y_true_bin[:, i],
                                     np.array([score[i] for score in y_scores]))  # AP (Area under PR curve)
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {100 * ap:.6f}%)')

    # 设置图像的标签、标题和图例
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True)

    # 优化图像布局
    plt.tight_layout()

    # 显示图像
    plt.show()
