from dataset import SeqDataset
from torch.utils import data
from net import StructurePrediction
import torch.optim as optim
from tqdm import tqdm
import torch
import re
import os
import numpy as np
from test import RiNA


def load_model(model_dir):
    # 假设你的模型文件存储在'pre_point/'目录下
    model_pattern = re.compile(r'model_epoch_\d+\.pth')  # 正则表达式匹配模型文件
    # 找到所有匹配的模型文件路径
    model_files = [f for f in os.listdir(model_dir) if model_pattern.match(f)]
    print("找到的模型文件：", model_files)
    # 如果没有找到任何模型文件
    if not model_files:
        print("没有找到任何模型文件，从0开始训练！")
        return None
    else:
        # 找到最后一次训练的文件
        max_num_file = max(model_files, key=lambda x: int(re.search(r'\d+', x).group()))
        last_path = os.path.join(model_dir, max_num_file)
        # max_num = int(re.search(r'\d+', max_num_file).group())
        print(f"找到并准备加载模型: {last_path}, 继续训练")

        return last_path


def val(val_data_loader, model, epoch):
    print('Star validation：')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    res = {'epoch': [], 'f1': [], 'precision': [], 'recall': []}
    with torch.no_grad():
        for batch in tqdm(val_data_loader, total=len(val_data_loader)):

            val_input1 = batch["input1"].to(device)
            val_contact = batch["contact"].to(device)  # [batch, seq_len, seq_len]
            val_seq = batch["sequence"][0]
            val_one_hot = batch["one_hot"].to(device)
            val_dpcp = batch["dpcp"].to(device)
            val_ncp = batch["ncp"].to(device)
            val_pred = model(val_one_hot, val_dpcp, val_ncp, val_input1)
            val_pred = val_pred.squeeze(0)
            val_contact = val_contact.squeeze(0)
            f1, precision, recall = RiNA(val_pred, val_contact, val_seq)
            res['f1'].append(f1)
            res['precision'].append(precision)
            res['recall'].append(recall)
    average_f1 = sum(RiNA_res['f1']) / len(RiNA_res['f1'])
    average_precision = sum(res['precision']) / len(res['precision'])
    average_recall = sum(res['recall']) / len(res['recall'])
    print('res:', 'average_f1:', average_f1, 'average_precision', average_precision, 'average_recall', average_recall)
    with open('res.txt', 'a') as file:
        # 将列表中的每个元素转换为字符串，并在每个元素后加上换行符'\n'，然后写入文件
        file.write(f"{epoch}, {average_f1}, {average_precision}, {average_recall}\n")
    return average_f1


def train(model, train_data_loader, criterion, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('start training...')
    # There are three steps of training
    losses = []
    model.train()
    # 定义 ANSI 转义码
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    loop = tqdm(train_data_loader, total=len(train_data_loader), leave=True,
                bar_format=f'{WHITE}{{l_bar}}{WHITE}{{bar}}{WHITE}{{r_bar}}{ENDC}')
    for batch in loop:
        """x, length, fm, interaction_prior, input1"""

        input1 = batch["input1"].to(device)
        one_hot = batch["one_hot"].to(device)
        dpcp = batch["dpcp"].to(device)
        ncp = batch["ncp"].to(device)

        contacts = batch["contact"].to(device)  # [batch, seq_len, seq_len]
        pred = model(one_hot, dpcp, ncp, input1)
        loss = criterion(pred, contacts)
        losses.append(loss.item())
        # Optimize the pre_point
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch [{epoch}/100]')
        loop.set_postfix(loss=loss.item())
        # torch.cuda.empty_cache()
    with open('loss.txt', 'a') as file:
        # 将列表中的每个元素转换为字符串，并在每个元素后加上换行符'\n'，然后写入文件
        file.write(f"{epoch}, {np.mean(losses)}\n")
    print('Training log: epoch: {}, loss: {}'.format(epoch, np.mean(losses)))


def main(train_data, val_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = 0
    epoch_all = 200
    best_f1 = 0
    no_improvement_count = 0
    max_no_improvement = 30
    train_dataset = SeqDataset(data_path=train_data)
    train_data_loder = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_dataset = SeqDataset(data_path=val_data)
    val_data_loder = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(len(train_data_loder), len(val_data_loder))

    model = StructurePrediction()
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)

    if load_model('pre_point/net/'):
        model_pth = load_model('pre_point/net/')
        checkpoint = torch.load(model_pth)
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 加载学习率调度器状态
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # 加载其他信息，如epoch数和损失
        start_epoch = checkpoint['epoch']

    # contact_net.to(device)
    for epoch in range(start_epoch, epoch_all):
        train(model, train_data_loder, criterion, optimizer, epoch)
        RiNA_f1 = val(val_data_loder, model, epoch+1)
        if RiNA_f1 > best_f1:
            best_f1 = RiNA_f1
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        print('best_f1:', best_f1)
        if no_improvement_count >= max_no_improvement:
            print(f"No improvement for {max_no_improvement} epochs. Stopping training.")
            break

        # 在每个epoch结束时保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }, f'pre_point/net8/model_epoch_{epoch + 1}.pth')
        scheduler.step()


if __name__ == "__main__":

    train_datapath = "data/ArchiveII/train.csv"
    val_datapath = "data/ArchiveII/valid.csv"
    main(train_datapath, val_datapath)

