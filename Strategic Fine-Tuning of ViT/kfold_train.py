import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
import random


def cross_validate(args, train_images_path, train_images_label, device):
    """
    实现 K 折交叉验证 (5折)
    """
    kfold = KFold(n_splits=args.k, shuffle=True, random_state=42)  # 5折交叉验证

    best_val_acc = 0.0  # 记录最好的验证准确率
    # 为每一折创建保存图像的文件夹
    if not os.path.exists("plots"):
        os.makedirs("plots")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_images_path)):
        print(f"\nFold {fold + 1}/{args.k} - Training...")

        # 根据KFold的划分获取train和val的索引
        train_images_fold = [train_images_path[i] for i in train_idx]
        train_labels_fold = [train_images_label[i] for i in train_idx]
        val_images_fold = [train_images_path[i] for i in val_idx]
        val_labels_fold = [train_images_label[i] for i in val_idx]

        # 添加随机噪声的自定义函数
        def add_random_noise(img):
            """在图像上添加随机噪声（例如高斯噪声）"""
            if random.random() < 0.5:  # 50%的概率添加噪声
                noise = torch.randn_like(img) * 0.05  # 控制噪声的强度
                img = img + noise
                img = torch.clamp(img, 0.0, 1.0)  # 确保图像值在0-1之间
            return img

        # 创建数据集和数据加载器
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机水平翻转
                                         transforms.RandomHorizontalFlip(),

                                         # 旋转，范围设置为-5°到5°
                                         transforms.RandomRotation(degrees=5),
                                         # 随机平移，最大平移0.1倍图像的宽度和高度
                                         transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                         # 随机裁剪并加零填充
                                         transforms.RandomCrop(size=224, padding=4),
                                         # 随机调整亮度、对比度、饱和度
                                         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),

                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda img: add_random_noise(img)),
                                         transforms.Normalize([0.5, 0.5, 0.5],
                                                              [0.5, 0.5, 0.5])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])
        }


        train_dataset = MyDataSet(images_path=train_images_fold,
                                  images_class=train_labels_fold,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_path=val_images_fold,
                                images_class=val_labels_fold,
                                transform=data_transform["val"])

        # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 2,8])  # number of workers
        nw=0
        print("number of workers: ",nw)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=nw, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=nw, pin_memory=True)

        # 创建模型
        model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

        if args.weights != "":
            assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
            weights_dict = torch.load(args.weights, map_location=device)
            del_keys = ['head.weight', 'head.bias'] if model.has_logits else \
                ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        # 冻结层
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)

        # 初始化优化器
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)

        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5E-5)
        # optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=5E-5)
        # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=5E-5)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5E-5)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x:
        ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf)

        # 用于记录每折的训练过程
        fold_results_file = f"fold_{fold + 1}_results.txt"

        with open(fold_results_file, "w") as f:
            f.write(f"Results for Fold {fold + 1}:\n")
            f.write("-" * 30 + "\n")

        # 存储每一折的损失和准确率
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # 训练和验证
        best_fold_val_acc = 0.0


        last_val_acc = 0.0  # 记录最后一轮的验证准确率
        last_val_precision = 0.0
        last_val_recall = 0.0
        last_val_F1_score = 0.0
        start_time = time.time()


        for epoch in range(args.epochs):
            # 记录训练时间
            epoch_start_time = time.time()
            # 训练一个epoch
            train_loss, train_acc, train_precision, train_recall, train_F1_score = train_one_epoch(model=model,
                                                                                                   optimizer=optimizer,
                                                                                                   data_loader=train_loader,
                                                                                                   device=device,
                                                                                                   epoch=epoch)
            # validate
            val_loss, val_acc, val_precision, val_recall, val_F1_score = evaluate(model=model,
                                                                                  data_loader=val_loader,
                                                                                  device=device,
                                                                                  epoch=epoch)

            last_val_acc = val_acc  # 更新最后的验证准确率
            last_val_precision = val_precision  # 更新最后的精确率
            last_val_recall = val_recall  # 更新最后的召回率
            last_val_F1_score = val_F1_score  # 更新最后的F1-score

            torch.save(model.state_dict(), f"weights/last_fold{fold + 1}.pth")

            # 更新最优验证准确率
            if val_acc > best_fold_val_acc:
                best_fold_val_acc = val_acc
                torch.save(model.state_dict(), f"weights/best_fold{fold + 1}.pth")
                print(f"Fold {fold + 1}, Epoch {epoch}: Best model saved with validation accuracy: {val_acc:.4f}")

            # 更新学习率
            scheduler.step()
            # 计算每个epoch的时间
            epoch_time = time.time() - epoch_start_time

            # 将结果写入文件
            # write into txt
            with open(fold_results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {train_loss:.4f}\n" \
                             f"train_acc: {train_acc:.4f}\n" \
                             f"train_precision: {train_precision:.4f}\n" \
                             f"train_recall: {train_recall:.4f}\n" \
                             f"train_F1_score: {train_F1_score:.4f}\n" \
                             f"val_loss: {val_loss:.4f}\n" \
                             f"val_acc: {val_acc:.4f}\n" \
                             f"val_precision: {val_precision:.4f}\n" \
                             f"val_recall: {val_recall:.4f}\n" \
                             f"val_F1_score: {val_F1_score:.4f}\n" \
                             f"lr: {args.lr:.6f}\n" \
                             f"epoch_time: {epoch_time:.2f}s\n"
                f.write(train_info + "\n\n")

            # 保存每个epoch的损失和准确率
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)


        print(f"Best validation accuracy for fold {fold + 1}: {best_fold_val_acc:.4f}")



        # 记录最好的验证集准确率
        if best_fold_val_acc > best_val_acc:
            best_val_acc = best_fold_val_acc


        # 训练结束后记录训练时长和最终的准确率信息
        total_training_time = time.time() - start_time
        with open(fold_results_file, "a") as f:
            final_info = f"\nTraining Complete\n" \
                         f"Best validation accuracy: {best_val_acc:.4f}\n" \
                         f"Last validation accuracy: {last_val_acc:.4f}\n" \
                         f"Last validation precision: {last_val_precision:.4f}\n" \
                         f"Last validation recall: {last_val_recall:.4f}\n" \
                         f"Last validation F1_score: {last_val_F1_score:.4f}\n" \
                         f"Total training time: {total_training_time:.2f}s\n"
            f.write(final_info)

        # 设置全局字体为 Times New Roman，字号为8pt
        plt.rcParams.update({
            'font.family': 'Times New Roman',  # 字体
            'font.size': 12,  # 字号
            'axes.labelsize': 12,  # 坐标轴标签的字体大小
            'axes.titlesize': 12,  # 标题的字体大小
            'xtick.labelsize': 12,  # x轴刻度标签的字体大小
            'ytick.labelsize': 12,  # y轴刻度标签的字体大小
            'legend.fontsize': 12,  # 图例字体大小
            'figure.figsize': (6, 4),  # 图形大小，可以根据需要调整
        })

        # 绘制训练和验证损失
        plt.plot(range(args.epochs), train_losses, label="Train Loss")
        plt.plot(range(args.epochs), val_losses, label="Val Loss")
        plt.title(f"Fold {fold + 1} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./plots/fold_{fold + 1}_loss.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制训练和验证准确率
        plt.plot(range(args.epochs), train_accuracies, label="Train Accuracy")
        plt.plot(range(args.epochs), val_accuracies, label="Val Accuracy")
        plt.title(f"Fold {fold + 1} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # 设置边距为0pt
        plt.savefig(f"./plots/fold_{fold + 1}_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()


    print(f"\nBest validation accuracy across all folds: {best_val_acc:.4f}")
    return best_val_acc


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    if not os.path.exists("weights"):
        os.makedirs("weights")

    tb_writer = SummaryWriter()
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # 使用交叉验证进行训练和验证
    best_val_acc = cross_validate(args, train_images_path, train_images_label, device)

    print(f"Best validation accuracy across all folds: {best_val_acc:.4f}")
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation')  # 这里设置为5折交叉验证
    parser.add_argument('--data-path', type=str,
                        default=r"Axial")
    parser.add_argument('--model-name', default='', help='create model name')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights',
                        type=str,
                        default='./vit_base_patch16_224_in21k.pth',
                        # default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='Device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
