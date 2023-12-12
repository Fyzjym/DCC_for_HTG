# from models.model import VATr
from models.model import TRGAN
import argparse
import torch
import collections
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

class FolderDataset:
    def __init__(self, folder_path, num_examples=15):
        folder_path = Path(folder_path)
        self.imgs = list(folder_path.iterdir())
        self.transform = get_transform(grayscale=True)
        self.num_examples = num_examples

    def __len__(self):
        return len(self.imgs)

    def sample_style(self):
        random_idxs = np.random.choice(len(self.imgs), self.num_examples, replace=False)
        imgs = [Image.open(self.imgs[idx]).convert('L') for idx in random_idxs]
        imgs = [img.resize((img.size[0] * 32 // img.size[1], 32), Image.Resampling.BILINEAR) for img in imgs]
        imgs = [np.array(img) for img in imgs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
        }
        return item



def load_checkpoint(model, checkpoint):
    if not isinstance(checkpoint, collections.OrderedDict):
        checkpoint = checkpoint['model']
    old_model = model.state_dict()
    if len(checkpoint.keys()) == 241:  # default
        counter = 0
        for k, v in checkpoint.items():
            if k in old_model:
                old_model[k] = v
                counter += 1
            elif 'netG.' + k in old_model:
                old_model['netG.' + k] = v
                counter += 1

        ckeys = [k for k in checkpoint.keys() if 'Feat_Encoder' in k]
        okeys = [k for k in old_model.keys() if 'Feat_Encoder' in k]
        for ck, ok in zip(ckeys, okeys):
            old_model[ok] = checkpoint[ck]
            counter += 1
        assert counter == 241
        checkpoint_dict = old_model
    else:
        assert len(old_model) == len(checkpoint)
        checkpoint_dict = {k2: v1 for (k1, v1), (k2, v2) in zip(checkpoint.items(), old_model.items()) if
                           v1.shape == v2.shape}
    assert len(old_model) == len(checkpoint_dict)
    model.load_state_dict(checkpoint_dict, strict=False)
    return model


class FakeArgs:
    feat_model_path = 'files/resnet_18_pretrained.pth'
    label_encoder = 'default'
    save_model_path = 'saved_models'
    dataset = 'IAM'
    english_words_path = 'files/english_words.txt'
    wandb = False
    no_writer_loss = False
    writer_loss_weight = 1.0
    no_ocr_loss = False
    img_height = 32
    resolution = 16
    batch_size = 32
    num_workers = 4
    num_epochs = 100
    lr = 0.0001
    num_examples = 15
    is_seq = True
    is_kld = False
    tn_hidden_dim = 512
    tn_nheads = 8
    tn_dim_feedforward = 512
    tn_dropout = 0.1
    tn_enc_layers = 3
    tn_dec_layers = 3
    alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    special_alphabet = 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
    query_input = 'unifont'
    query_linear = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(alphabet)
    num_writers = 339  # 339 for IAM, 283 for CVL
    g_lr = 0.00005
    d_lr = 0.00005
    w_lr = 0.00005
    ocr_lr = 0.00005
    add_noise = True
    all_chars = False

class strLabelConverter_vatr(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''
        length = []
        result = []
        results = []
        for item in text:
            if isinstance(item, bytes): item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
            results.append(result)
            result = []

        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(text) for text in results], batch_first=True), torch.IntTensor(length), None

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class TRGAN_writer:
    def __init__(self, checkpoint_path, args=FakeArgs()):
        self.model = TRGAN()
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        load_checkpoint(self.model, checkpoint)
        self.model.eval()
        self.style_dataset = None
        alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
        self.netconverter = strLabelConverter_vatr(alphabet)
        self.resolution = 16

    def set_style_folder(self, style_folder, num_examples=15):
        self.style_dataset = FolderDataset(style_folder, num_examples=num_examples)

    @torch.no_grad()
    def generate(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.style_dataset is None:
            raise Exception('Style is not set')

        gap = np.ones((32, 16))
        fakes = []
        for i, text in enumerate(texts, 1):
            # print(f'[{i}/{len(texts)}] Generating for text: {text}')
            style = self.style_dataset.sample_style()
            style_imgs = style['simg'].unsqueeze(0).to(DEVICE)
            # print(text.split())
            text_encode, len_text, encode_pos = self.netconverter.encode(text.split())
            text_encode = text_encode.to(DEVICE).unsqueeze(0)

            fake = self.model._generate_fakes(style_imgs, text_encode, len_text, encode_pos)


            fake = np.concatenate(sum([[img, gap] for img in fake], []), axis=1)[:, :-16]
            # 改变尺寸
            # new_height = 64
            # new_width = int(fake.shape[1] * (new_height / fake.shape[0]))
            # fake = cv2.resize(fake, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            fake = (fake * 255).astype(np.uint8)
            fakes.append(fake)

        return fakes

wid_60wid_select_list = [ '000','005','006','007','008','012','013','017','018','019',
                          '021','025','026','029','036','038','042','043','049','054',
                          '060','063','066','081','085','090',
                          '211','216','217','223','227','232','233','239','242','246',
                          '247','259','274','230','225','265',
                          '328','332','333','334','337','338','339','340','342','343',
                          '551','635','640','644','654','663','668','670',
                          ]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--style-folder", default='files/style_samples/00', type=str)
    # parser.add_argument("-t", "--text", default='That\'s one small step for man, one giant leap for mankind ΑαΒβΓγΔδ',
    #                     type=str)
    # parser.add_argument("--text-path", default=None, type=str, help='Path to text file with texts to generate')
    # parser.add_argument("-c", "--checkpoint", default='files/vatr.pth', type=str)
    # parser.add_argument("-o", "--output-folder", default='files/output', type=str)  # Output folder
    parser.add_argument("--add_noise", action='store_true')
    parser.add_argument("--epoch")
    parser.add_argument("--exp_name")
    args = parser.parse_args()
    epoch = args.epoch
    exp_name = args.exp_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # @ym
    model_name = 'model{}'.format(epoch)
    model_path = './saved_models/{}/'.format(exp_name) + model_name + '.pth'


    out_put_file = "./saved_images/{}/TestSet/".format(model_name)
    style_file = "/home/WeiHongxi/Node95/Ym/data/words_wid/"


    for iwid in test_wid:

        # wid
        text_path = "/home/WeiHongxi/Node95/Ym/Project_20230709_VATr/CommData/eachWidWords/{}_words.lst".format(iwid)
        if text_path is not None:
            with open(text_path, 'r') as f:
                text_lines = f.read()
        text_split_lines = text_lines.splitlines()


        output_folder = out_put_file + '{}/'.format(iwid)
        style_folder = style_file + '{}'.format(iwid)
    # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fake_args = FakeArgs()
        fake_args.add_noise = args.add_noise
        writer = TRGAN_writer(model_path, fake_args)
        writer.set_style_folder(style_folder)
        print("run @ :{}".format(style_folder))
        for i, text in enumerate(tqdm(text_split_lines)):
            # Generate the image
            fakes = writer.generate(text)

            for j, fake in enumerate(fakes):
                # Build the output filename using style, word id, and label
                output_filename = f"{os.path.basename(style_folder)}_{i:04d}_{text.replace(' ', '_')}.png"
                output_path = os.path.join(output_folder, output_filename)
                # fake = np.array(fake)
                # Save the image
                cv2.imwrite(output_path, fake)

    print('Done')

