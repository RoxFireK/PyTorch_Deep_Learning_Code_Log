import numpy as np
import pandas as pd
import os


def one_hot_encode_along_channel_axis(sequence):
    """将DNA序列编码为one-hot矩阵 (4, seq_len)"""
    to_return = np.zeros((4, len(sequence)), dtype=np.int8)
    for i, char in enumerate(sequence):
        if char in 'Aa':
            char_idx = 0
        elif char in 'Cc':
            char_idx = 1
        elif char in 'Gg':
            char_idx = 2
        elif char in 'Tt':
            char_idx = 3
        elif char in 'Nn':
            continue
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        to_return[char_idx, i] = 1
    return to_return


def load_and_preprocess_data(excel_path):
    """读取Excel并返回one-hot编码后的训练/验证/测试数据"""
    df = pd.read_excel(excel_path)
    print('genes number:', df.shape[0])

    # 训练集
    df_train = df[df['dataset'] == 'train']
    y_train = np.log2(df_train['TPM'].values + 1)
    train_data = np.array([one_hot_encode_along_channel_axis(str(i).strip())
                           for i in df_train['sequence'].values])

    # 验证集
    df_valid = df[df['dataset'] == 'valid']
    y_valid = np.log2(df_valid['TPM'].values + 1)
    valid_data = np.array([one_hot_encode_along_channel_axis(str(i).strip())
                           for i in df_valid['sequence'].values])

    # 测试集
    df_test = df[df['dataset'] == 'test']
    y_test = np.log2(df_test['TPM'].values + 1)
    test_data = np.array([one_hot_encode_along_channel_axis(str(i).strip())
                          for i in df_test['sequence'].values])

    return train_data, y_train, valid_data, y_valid, test_data, y_test