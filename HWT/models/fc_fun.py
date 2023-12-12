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
    train_wid = ['670', '667', '343', '000', '344', '009', '005', '551', '635', '269', '214', '327', '174', '132',
                 '128', '118', '239', '275', '062', '330', '154', '636', '289', '102', '324', '254', '333', '092',
                 '278', '332', '270', '124', '064', '544', '227', '192', '213', '151', '026', '097', '320', '217',
                 '250', '125', '085', '334', '147', '234', '336', '243', '150', '202', '100', '139', '060', '109',
                 '089', '164', '152', '274', '197', '160', '025', '264', '087', '116', '155', '016', '044', '068',
                 '162', '219', '095', '117', '033', '272', '088', '233', '549', '006', '548', '671', '613', '605',
                 '340', '658', '338', '341', '337', '669', '329', '010', '586', '335', '342', '247', '625', '138',
                 '107', '051', '058', '257', '093', '123', '081', '228', '140', '028', '020', '142', '166', '126',
                 '540', '246', '130', '660', '531', '640', '582', '121', '061', '235', '133', '113', '127', '559',
                 '136', '090', '032', '265', '114', '056', '184', '273', '112', '145', '220', '079', '255', '259',
                 '249', '178', '071', '229', '664', '663', '339', '039', '045', '037', '042', '024', '149', '131',
                 '063', '604', '527', '236', '328', '521', '153', '193', '103', '171', '108', '111', '216', '248',
                 '059', '163', '222', '070', '129', '252', '050', '212', '077', '650', '652', '541', '666', '040',
                 '065', '104', '099', '260', '110', '211', '018', '221', '013', '626', '643', '529', '638', '627',
                 '001', '662', '595', '084', '165', '144', '533', '119', '054', '034', '223', '621', '567', '014',
                 '008', '137', '011', '002', '649', '225', '519', '080', '323', '266', '268', '515', '639', '035',
                 '135', '156', '230', '019', '027', '041', '096', '055', '242', '232', '141', '167', '301', '094',
                 '047', '091', '053', '074', '083', '143', '007', '003', '659', '648', '651', '611', '661', '642',
                 '012', '158', '004', '069', '046', '215', '043', '082', '290', '267', '653', '326', '238', '036',
                 '224', '017', '237', '076', '310', '226', '262', '134', '632', '665', '574', '628', '022', '066',
                 '098', '218', '157', '048', '240', '624', '122', '637', '612', '241', '120', '511', '317', '668',
                 '606', '256', '067', '161', '073', '148', '244', '105', '231', '030', '159', '210', '115', '086',
                 '655', '644', '641', '654', '176', '023', '052', '031', '258', '261', '182', '029', '106', '038',
                 '049', '331', '575', '645', '576', '566', '251', '245', '146', '263', '253', '075', '072', '078',
                 '015', '647', '021']

    test_wid = ['552', '280', '600', '191', '285', '560', '318', '580', '288', '173', '563', '305', '508', '537', '207',
                '169', '287', '545', '325', '299', '199', '295', '209', '203', '175', '177', '204', '298', '602', '616',
                '291', '634', '585', '518', '512', '590', '571', '622', '293', '583', '601', '281', '205', '578', '565',
                '615', '302', '555', '208', '553', '556', '610', '517', '584', '526', '587', '190', '198', '170', '181',
                '183', '188', '514', '591', '546', '536', '596', '516', '286', '194', '579', '547', '277', '315', '562',
                '180', '539', '542', '557', '614', '279', '313', '309', '206', '530', '617', '588', '292', '525', '620',
                '630', '550', '509', '195', '308', '510', '561', '581', '594', '569', '282', '543', '186', '513', '321',
                '179', '538', '593', '187', '276', '296', '322', '633', '577', '283', '300', '201', '200', '189', '631',
                '573', '589', '629', '532', '523', '570', '592', '168', '534', '618', '603', '522', '564', '314', '304',
                '297', '316', '598', '528', '619', '303', '568', '319', '172', '312', '185', '599', '608', '196', '609',
                '572', '506', '597', '520', '623', '524', '607', '558', '554', '535', '307']

    def cal_sfa(wid):
        n_count = wid_count_dict[wid]
        n_max = max(wid_count_dict.values())
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
        j = int(j)
        wid = train_wid[j]
        # tau_batch.append(tau[label[j].int()])
        tau_batch.append(cal_sfa(wid))


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
        print(label[j])
        print(tau[label[j].int()])

    print(tau)
    print(label)
    print(tau_batch)

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



