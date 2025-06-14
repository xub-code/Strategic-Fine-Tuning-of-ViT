import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from vit_model import vit_base_patch16_224_in21k as create_model
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
import matplotlib
import random

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 检查当前目录下是否存在名为 "weights" 的文件夹，如果不存在，则创建该文件夹。
    if os.path.exists("weights") is False:
        os.makedirs("weights")

    # 创建 logs 文件夹用来保存训练结果
    log_folder = './logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    tb_writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5],
                                                        [0.5, 0.5, 0.5])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,2, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    train_steps = len(train_loader)


    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    val_steps = len(validate_loader)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)




    num_classes=5
    net = create_model(num_classes=num_classes, has_logits=False).to(device)


    # load pretrain weights
    model_weight_path = "vit_base_patch16_224_in21k.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    if model_weight_path != "":
        assert os.path.exists(model_weight_path), "weights file: '{}' not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if net.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(net.load_state_dict(weights_dict, strict=False))

    freeze_layers = False # 这是一个控制是否冻结的标志
    if freeze_layers:
        for name, param in net.named_parameters():  # 遍历模型的所有参数
            if "head" not in name and "pre_logits" not in name:  # 如果参数不属于head或pre_logits层
                param.requires_grad = False  # 冻结该层的权重，避免更新
            else:
                print(f"Training {name}")  # 如果是head或pre_logits层，继续训练这些层

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    # params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)
    optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.01)

    # 用来保存训练以及验证过程中信息
    results_file = os.path.join(log_folder, "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    epochs = 50
    best_acc = 0.0

    ########################
    # 存储每个epoch的训练损失和准确率
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    ########################

    for epoch in range(epochs):
        # train
        net.train()

        running_train_loss = 0.0
        running_train_acc = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            train_images, train_labels = data

            optimizer.zero_grad()
            logits = net(train_images.to(device))
            loss = loss_function(logits, train_labels.to(device))
            loss.backward()
            optimizer.step()

            train_acc = torch.max(logits, dim=1)[1]
            running_train_acc += torch.eq(train_acc, train_labels.to(device)).sum().item()

            # print statistics
            running_train_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] train_loss:{:.3f} ".format(epoch + 1, epochs, loss.item())

        train_loss = running_train_loss / train_steps
        train_accurate = running_train_acc / train_num
        train_acc_list.append(train_accurate)
        train_loss_list.append(train_loss)

        # validate
        net.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0  # accumulate accurate number / epoch

        # 用于计算Precision, Recall, F1-score的变量
        TP = [0] * num_classes  # 每个类的真正例数
        FP = [0] * num_classes  # 每个类的假正例数
        FN = [0] * num_classes  # 每个类的假负例数
        TN = [0] * num_classes  # 每个类的真负例数


        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))

                # val_loss
                logits = net(val_images.to(device))  # 不用辅助分类器
                loss = loss_function(logits, val_labels.to(device))
                # print statistics
                running_val_loss += loss.item()

                val_acc = torch.max(outputs, dim=1)[1]
                running_val_acc += torch.eq(val_acc, val_labels.to(device)).sum().item()

                # 更新TP, FP, FN, TN
                for i in range(len(val_labels)):
                    true_label = val_labels[i].item()
                    pred_label = val_acc[i].item()

                    if true_label == pred_label:
                        TP[true_label] += 1
                    else:
                        FP[pred_label] += 1
                        FN[true_label] += 1


                val_bar.desc = "val epoch[{}/{}] val_loss:{:.3f} ".format(epoch + 1, epochs, loss.item())

        val_loss = running_val_loss / val_steps
        val_accurate = running_val_acc / val_num
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accurate)

        # 计算精确率、召回率和F1-score（手动计算）
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for i in range(num_classes):
            precision = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
            recall = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        # 计算精确率、召回率和F1-score的平均值
        avg_precision = total_precision / num_classes
        avg_recall = total_recall / num_classes
        avg_f1_score = total_f1 / num_classes

        # 输出训练和验证的损失、准确率以及精确率、召回率和F1-score
        epoch_result = f"Epoch [{epoch + 1}/{epochs}] - " \
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, " \
                       f"Train Accuracy: {train_accurate:.4f}, Val Accuracy: {val_accurate:.4f}, " \
                       f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1_score:.4f}\n"

        # 保存每个epoch的训练结果
        with open(results_file, 'a') as f:
            f.write(epoch_result)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # 向 TensorBoard 的 SummaryWriter 中添加一个标量（scalar）数值数据。
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_accurate, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_accurate, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # 将这些指标写入TensorBoard
        tb_writer.add_scalar("val_precision", avg_precision, epoch)
        tb_writer.add_scalar("val_recall", avg_recall, epoch)
        tb_writer.add_scalar("val_F1_score", avg_f1_score, epoch)

        # 输出训练和验证的损失、准确率以及精确率、召回率和F1-score
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Accuracy: {train_accurate:.4f}, Val Accuracy: {val_accurate:.4f}, "
              f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1_score:.4f}")

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch_result = epoch_result
            torch.save(net.state_dict(), f"weights/best.pth")



    # 在循环结束后保存模型权重
    torch.save(net.state_dict(), f"weights/last.pth")



    # 设置全局字体为 Times New Roman
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    # 设置字体大小为 8pt
    matplotlib.rcParams['font.size'] = 8
    # 设置图表大小 (宽度 17cm, 高度按比例设置)
    # 1 inch = 2.54 cm, 所以 17cm = 17/2.54 = 6.69 inches
    fig_width = 6.69  # inches
    fig_height = 6.69 * 0.6  # 高度根据宽度比例调整，假设是 0.6
    # 创建图形并设置大小
    plt.figure(figsize=(fig_width, fig_height))
    ############################

    # 绘制损失图
    plt.plot(train_loss_list, 'b-', label='Train Loss',linewidth=0.75)
    plt.plot(val_loss_list, 'r-', label='Validation Loss',linewidth=0.75)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_graph.png', dpi=300)
    plt.close()

    # 绘制准确率图
    plt.plot(train_acc_list, 'b-', label='Train Accuracy',linewidth=0.75)
    plt.plot(val_acc_list, 'r-', label='Validation Accuracy',linewidth=0.75)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_graph.png', dpi=300)
    plt.close()
    #############################

    # 保存最好的epoch的训练结果
    with open(os.path.join(log_folder, "best_epoch.txt"), 'w') as f:
        f.write(best_epoch_result)

    print('Finished Training')
    tb_writer.close()


if __name__ == '__main__':
    main()
