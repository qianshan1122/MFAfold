# from net4 import LanguagePrediction
from net8 import StructurePrediction
import torch
from tqdm import tqdm
from dataset4 import SeqDataset
from torch.utils import data
from postprocess import prob_mat_to_sec_struct
from utils import contact_f1, ss_f1, ss_precision, ss_recall
import statistics
import numpy as np
import pandas as pd


def matrix_to_dot_bracket(matrix):
    """
    将RNA配对矩阵转换为点括号图。

    参数:
    - matrix: 一个n x n的二维列表，代表RNA的配对信息。

    返回:
    - dot_bracket: 字符串，代表RNA结构的点括号图表示。
    """
    size = len(matrix)
    dot_bracket = ['.'] * size  # 初始化点括号图，全部设为未配对（点）

    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i][j] == 1:  # 如果找到配对
                dot_bracket[i] = '('  # 配对起始
                dot_bracket[j] = ')'  # 配对结束

    return ''.join(dot_bracket)


def val(val_data_loader, model):
    print('Star validation：')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    RiNA_res1 = {'id': [], 'f1': [], 'precision': [], 'recall': []}

    with torch.no_grad():
        for batch in tqdm(val_data_loader, total=len(val_data_loader)):
            val_input1 = batch["input1"].to(device)
            val_contact = batch["contact"].to(device)  # [batch, seq_len, seq_len]
            val_seq = batch["sequence"][0]
            val_one_hot = batch["one_hot"].to(device)
            val_dpcp = batch["dpcp"].to(device)
            val_ncp = batch["ncp"].to(device)
            val_id = batch["seqid"][0]

            val_pred = model(val_one_hot, val_dpcp, val_ncp, val_input1)
            result_no_train = ufold(val_pred, val_contact, val_one_hot, result_no_train)
            val_pred = val_pred.squeeze(0)
            val_contact = val_contact.squeeze(0)
            RiNA_f1, RiNA_precision, RiNA_recall = RiNA(val_pred, val_contact, val_seq)
            RiNA_res1['f1'].append(RiNA_f1)
            RiNA_res1['precision'].append(RiNA_precision)
            RiNA_res1['recall'].append(RiNA_recall)
            RiNA_res1['id'].append(val_id)

    average_f1 = sum(RiNA_res1['f1']) / len(RiNA_res1['f1'])
    average_precision = sum(RiNA_res1['precision']) / len(RiNA_res1['precision'])
    average_recall = sum(RiNA_res1['recall']) / len(RiNA_res1['recall'])
    print('RiNA_res 中位数:', statistics.median(RiNA_res1['f1']), 'average_f1:', average_f1, 'average_precision',
          average_precision, 'average_recall',average_recall)


def RiNA(pred, contact, seq):
    probs = torch.sigmoid(pred)
    if probs.dtype == torch.bfloat16:
        # Cast brain floating point into floating point
        probs = probs.type(torch.float16)
    probs = probs.detach().cpu().numpy()
    val_contact_cpu = contact.cpu().numpy()
    RiNA_pred = prob_mat_to_sec_struct(probs=probs, seq=seq, threshold=0.5)
    RiNA_precision = ss_precision(val_contact_cpu, RiNA_pred)
    RiNA_recall = ss_recall(val_contact_cpu, RiNA_pred)
    RiNA_f1 = ss_f1(val_contact_cpu, RiNA_pred)
    return RiNA_f1, RiNA_precision, RiNA_recall


def main(datapath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = SeqDataset(data_path=datapath)
    print("len(test_data):", len(test_data))
    test_data_loder = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    # pre_point = U_Net(img_ch=2)
    model = LanguagePrediction()
    model.to(device)

    model_pth = "pre_point/net_RNAstralign/best.pth"

    if torch.cuda.is_available():
        checkpoint = torch.load(model_pth, map_location=torch.device('cuda:0'))
    else:
        checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    val(test_data_loder, model)


if __name__ == "__main__":
    # test_datapath = "data/ArchiveII/test.csv"
    # test_datapath = "data/bprna/TS0.csv"
    test_datapath = "data/RNAstralign/test.csv"

    main(test_datapath)
