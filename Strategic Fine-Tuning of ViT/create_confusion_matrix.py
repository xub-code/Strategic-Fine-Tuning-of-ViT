import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils import read_split_data
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["Class", "Precision (%)", "Recall (%)", "Specificity (%)"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP) * 100, 3) if TP + FP != 0 else 0.0
            Recall = round(TP / (TP + FN) * 100, 3) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP) * 100, 3) if TN + FP != 0 else 0.0

            table.add_row([self.labels[i], Precision, Recall, Specificity])

        print(table)

    def plot(self):
        # 设置全局字体为 Times New Roman，字号为8pt
        plt.rcParams.update({
            'font.family': 'Times New Roman',  # 字体
            'font.size': 16,  # 字号
            'axes.labelsize': 16,  # 坐标轴标签的字体大小
            'axes.titlesize': 16,  # 标题的字体大小
            'xtick.labelsize': 16,  # x轴刻度标签的字体大小
            'ytick.labelsize': 16,  # y轴刻度标签的字体大小
            'legend.fontsize': 16,  # 图例字体大小
            'figure.figsize': (4, 4),  # 图形大小，可以根据需要调整
        })
        matrix = self.matrix


        print(matrix)

        # 绘制热图
        plt.imshow(matrix, cmap=plt.cm.Blues)
        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar并调整位置
        # 显示colorbar并调整位置
        cbar = plt.colorbar(fraction=0.046, pad=0.04)  # fraction 控制colorbar的宽度，pad控制与矩阵的间距
        cbar.ax.tick_params(labelsize=14)  # 设置colorbar刻度字体大小
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    data_path=r"Axial"
    print(data_path)
    _, _, val_images_path, val_images_label = read_split_data(data_path)

    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5],
                                                        [0.5, 0.5, 0.5])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])


    batch_size = 64
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)


    net = create_model(num_classes=3, has_logits=False).to(device)
    # load pretrain weights
    model_weight_path =r"./results/5_fold_aug_6_AdamW/weights/last_fold1.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=3, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

