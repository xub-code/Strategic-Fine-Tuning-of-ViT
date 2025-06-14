import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# from mmsegmentation.projects.pp_mobileseg.inference_onnx import build_session
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import time


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("weights_5fold") is False:
        os.makedirs("weights_5fold")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw=0 #windows
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers ubuntu
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']

        # 确保只有在键存在时才删除
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
            else:
                print(f"Warning: key '{k}' not found in the weights dictionary.")

        print(model.load_state_dict(weights_dict, strict=False))
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除不需要的权重
    #     del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #         else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    #     for k in del_keys:
    #         del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.AdamW(pg,lr=args.lr,weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 存储每个epoch的训练/验证损失和准确率
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_val_acc=0.0 # 记录验证集上最好的准确率
    last_val_acc=0.0 # 记录最后一轮的验证准确率

    val_precision1 = 0.0
    last_val_precision = 0.0

    val_recall1 = 0.0
    last_val_recall = 0.0

    val_F1_score1=0.0
    last_val_F1_score=0.0


    start_time=time.time()


    for epoch in range(args.epochs):
        # 记录训练开始时间
        epoch_start_time = time.time()
        # train
        train_loss, train_acc,train_precision, train_recall,train_F1_score  = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc,val_precision, val_recall,val_F1_score = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        last_val_acc=val_acc  # 更新最后的验证准确率
        last_val_precision = val_precision  # 更新最后的精确率
        last_val_recall = val_recall  # 更新最后的召回率
        last_val_F1_score = val_F1_score  # 更新最后的F1-score

        # 计算每个epoch的时间
        epoch_time = time.time() - epoch_start_time

        # write into txt
        with open(results_file, "a") as f:
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
                         f"lr: {args.lr:.6f}\n"\
                        f"epoch_time: {epoch_time:.2f}s\n"
            f.write(train_info + "\n\n")


        ##################################
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        ##################################

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # 向 TensorBoard 的 SummaryWriter 中添加一个标量（scalar）数值数据。
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)


        # 如果需要保存模型的完整结构和权重，可以直接保存模型对象。
        # torch.save(model,"./weights/model-{}.pth".format(epoch))
        # 模型的状态字典包含了模型中每一层的参数，是一种轻量级的保存方式。通常推荐使用这种方法，因为它不保存模型的计算图结构，节省空间且更加灵活。

        # 保存最好的权重（根据 val_acc）
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            val_precision1 = val_precision  # 保存精确率
            val_recall1 = val_recall  # 保存召回率
            val_F1_score1=val_F1_score # 保存val_F1_score


            torch.save(model.state_dict(), "weights_5fold/best.pth")
            print(f"Epoch {epoch}: Best model saved with validation accuracy: {val_acc:.4f}")

        # 保存每个epoch的权重（最后的权重）
        torch.save(model.state_dict(), "weights_5fold/last.pth")
        print(f"Epoch {epoch}: Last model saved with validation accuracy: {val_acc:.4f}")

    # 训练结束后记录训练时长和最终的准确率信息
    total_training_time = time.time() - start_time
    with open(results_file, "a") as f:
        final_info = f"\nTraining Complete\n" \
                     f"Best validation accuracy: {best_val_acc:.4f}\n" \
                     f"Last validation accuracy: {last_val_acc:.4f}\n" \
                     f"validation precision1: {val_precision1:.4f}\n" \
                     f"Last validation precision: {last_val_precision:.4f}\n" \
                     f"validation recall1: {val_recall1:.4f}\n" \
                     f"Last validation recall: {last_val_recall:.4f}\n" \
                     f"validation F1_score1: {val_F1_score1:.4f}\n" \
                     f"Last validation F1_score: {last_val_F1_score:.4f}\n" \
                     f"Total training time: {total_training_time:.2f}s\n"
        f.write(final_info)

    # 输出最终的最好的和最后的
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Last validation accuracy: {last_val_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f}s")

    ############################
    # 设置全局字体为 Times New Roman，字号为8pt
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 字体
        'font.size': 8,  # 字号
        'axes.labelsize': 8,  # 坐标轴标签的字体大小
        'axes.titlesize': 8,  # 标题的字体大小
        'xtick.labelsize': 8,  # x轴刻度标签的字体大小
        'ytick.labelsize': 8,  # y轴刻度标签的字体大小
        'legend.fontsize': 8,  # 图例字体大小
        'figure.figsize': (6, 4),  # 图形大小，可以根据需要调整
    })


    # 绘制损失图
    plt.plot(train_loss_list, 'b-', label='Train Loss')
    plt.plot(val_loss_list, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # 设置边距为0pt
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('loss_graph.png',dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制准确率图
    plt.plot(train_acc_list, 'b-', label='Train Accuracy')
    plt.plot(val_acc_list, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    # 设置边距为0pt
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('accuracy_graph.png', dpi=300, bbox_inches='tight')  # dpi参数控制分辨率，bbox_inches='tight'确保去除多余的边距
    plt.close()
    #############################

    print('Finished Training')
    tb_writer.close()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"Axial")

    '''
    “/”：表示根目录，在windows系统下表示某个盘的根目录，如“E:\”；
    “./”：表示当前目录；（表示当前目录时，也可以去掉“./”，直接写文件名或者下级目录）
    “../”：表示上级目录
    '''

    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights',
                        type=str,
                        # default='vit_base_patch16_224_in21k.pth',
                        default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
