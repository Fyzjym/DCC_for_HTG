import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import time
import os
import shutil
import math





def read_file(file_path='/home/WeiHongxi/Node95/Ym/Project_20230709_VATr/VATr_FCC_tau_a/corpora_english/wid_count.dict'):
    with open(file_path, 'r') as file:
        content = file.read()
    return eval(content)


# file_content = read_file(file_path)
# print(file_content['000'])


def fea_com(feature, label, num_classes, **kwargs):
    """
    gamma: the hyper-parameter for setting scaling factor tau.
    c_type: compression type:
                'edc' is equal difference compression.
    GAMMA: tau's paramter, FCC paepr: [0.1, 0.5, 1, 2, 3]
    """

    # c_type = 'edc'
    c_type = 'sfa'
    GAMMA = 1

    batch_size = feature.shape[0]
    wid_count_dict = read_file()
    # compressing feature
    if c_type == 'edc':
        new_features = equal_diff_compress(batch_size, feature, label, num_classes, GAMMA)
    elif c_type == 'sfa':
        new_features = soft_feat_compress_tau_a(batch_size, feature, label, num_classes, GAMMA, wid_count_dict)
        # print(batch_size)
        # print(feature.shape)
        # print(label)
        # print(num_classes)
        # print(GAMMA)

    else:
        raise Exception('Error compression type.')

    return new_features

def soft_feat_compress_tau_a(n, feature, label, num_classes, gamma, wid_count_dict):
    """
    todo: 多数类 tau较大
    :param n: batch_size
    :param feature:
    :param label: wid的索引 index，可使用 train_wid[index]获得真正的wid
    :param num_classes:
    :param gamma:
    :return:
    """

    # 339  index image count
    train_wid = [53, 263, 68, 303, 106, 481, 481, 236, 49, 62, 243, 481, 2839, 138, 482, 511, 193, 143, 109, 140, 354, 418, 88, 149, 255, 217, 237, 290, 514, 481, 193, 464, 481, 108, 62, 107, 49, 53, 245, 54, 143, 110, 144, 504, 105, 46, 160, 54, 169, 204, 525, 63, 96, 249, 106, 109, 53, 516, 239, 59, 192, 481, 108, 116, 47, 100, 101, 69, 198, 81, 193, 483, 50, 202, 201, 53, 57, 47, 137, 30, 43, 153, 101, 50, 65, 511, 45, 225, 208, 211, 51, 120, 38, 227, 441, 582, 57, 205, 109, 182, 477, 187, 46, 56, 87, 108, 96, 254, 149, 65, 87, 43, 43, 184, 161, 99, 110, 99, 87, 161, 32, 58, 112, 118, 50, 45, 42, 52, 171, 410, 55, 86, 100, 46, 42, 129, 52, 95, 58, 54, 33, 48, 49, 94, 41, 62, 212, 161, 50, 48, 49, 66, 47, 92, 37, 52, 481, 121, 117, 57, 97, 65, 217, 191, 36, 55, 43, 52, 235, 105, 58, 48, 49, 49, 49, 44, 54, 209, 204, 321, 106, 49, 49, 36, 66, 294, 102, 516, 59, 58, 29, 108, 53, 57, 49, 49, 55, 131, 41, 162, 106, 43, 121, 51, 43, 42, 92, 144, 41, 51, 51, 99, 137, 164, 43, 42, 45, 45, 52, 56, 56, 47, 174, 107, 55, 82, 45, 45, 87, 76, 88, 58, 137, 171, 47, 37, 108, 73, 45, 114, 41, 39, 62, 46, 40, 54, 57, 53, 42, 43, 69, 53, 95, 36, 52, 62, 55, 45, 41, 72, 54, 49, 49, 44, 58, 33, 88, 49, 49, 38, 28, 22, 50, 50, 47, 63, 44, 60, 48, 47, 49, 53, 46, 118, 267, 579, 82, 7, 64, 60, 114, 46, 60, 105, 44, 37, 257, 204, 110, 62, 89, 305, 100, 108, 45, 38, 52, 42, 61, 205, 178, 140, 47, 49, 52, 53, 39, 48, 48, 64, 251, 42, 61, 51, 112, 237, 302, 51, 73, 53, 120, 115, 253, 56, 46, 56, 130, 45, 94]



    def cal_sfa(wid_index):
        n_count = train_wid[wid_index]
        n_max = max(train_wid)
        return round((1 + math.sqrt((n_count/n_max))), 2)

    # setting scaling factor tau

    # tau = []
    # for k in range(num_classes):
    #     tau.append(round((1 + gamma - k * (gamma / num_classes)), 2))

    raw_shape = feature.shape
    # print(label)

    tau_batch = []
    # for j in range(n):
    for j in label.tolist():
        wid_index = int(j)
        # GANwriting 中 y 即是 wid，因此，不需要字典 index 索引wid
        # 已经证明：仍然使用index作为wid索引

        # tau_batch.append(tau[label[j].int()])

        tau_batch.append(cal_sfa(wid_index))


    tau_batch = torch.tensor(tau_batch).cuda()
    # print(tau_batch)
    tau_batch = tau_batch.view(n, 1)
    feature = feature.view(n, -1)

    new_features = torch.mul(feature, tau_batch)
    new_features = new_features.view(raw_shape)

    return new_features

def equal_diff_compress(n, feature, label, num_classes, gamma):
    """

    :param n: batch_size
    :param feature:
    :param label:
    :param num_classes:
    :param gamma:
    :return:
    """
    # setting scaling factor tau
    tau = []
    for k in range(num_classes):
        tau.append(round((1 + gamma - k * (gamma / num_classes)), 2))

    raw_shape = feature.shape

    tau_batch = []
    for j in range(n):
        tau_batch.append(tau[label[j].int()])
    #     print(label[j])
    #     print(tau[label[j].int()])
    #
    # print(tau)
    # print(label)
    # print(tau_batch)

    tau_batch = torch.tensor(tau_batch).cuda()
    # print(tau_batch)
    tau_batch = tau_batch.view(n, 1)
    feature = feature.view(n, -1)

    new_features = torch.mul(feature, tau_batch)
    new_features = new_features.view(raw_shape)

    return new_features


"""
def equal_diff_compress(n, feature, label, num_classes, gamma):
	'''
	This founction is an older version of FCC, which is slower and less performant compared to the above-mentioned version.
	'''

	tau = []
	for k in range(num_classes):
		tau.append((1 + gamma - k*round((gamma/num_classes),2)))

	new_features = []
	for i in range(n):
		new_features.append(feature[i]*tau[label[i]])

	#return feature
	return torch.stack(new_features)
"""



