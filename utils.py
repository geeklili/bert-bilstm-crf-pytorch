# -*- encoding: utf-8 -*-
import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from pytorch_pretrained_bert import BertTokenizer
from config import Config
logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained(Config.bert_model_dir)
VOCAB = ('<PAD>','[CLS]', '[SEP]',  'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')
# 代码中应用到的数据为医药命名体识别数据，已经处理成了BIO格式，其中B、I包含6个种类，分别为DSE(疾病和诊断），DRG（药品），OPS（手术），LAB（ 检验），PAT（解剖部位）、INF（检查）。
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS',
#         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2


class NerDataset(Dataset):
    def __init__(self, f_path):
        """将句子长度大于254的分成最长不超过254的小段，两头分别加上CLS和SEP，句子长度小于254的则直接加上CLS和SEP"""
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            if len(words) > MAX_LEN:
                # 先对句号分段
                word, tag = [], []
                for char, t in zip(words, tags):
                    
                    if char != '。':
                        if char != '\ue236':   # 测试集中有这个字符
                            word.append(char)
                            tag.append(t)
                    else:
                        sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                        tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                        word, tag = [], []            
                # 最后的末尾
                if len(word):
                    sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                    tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                    word, tag = [], []
            else:
                sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li = sents, tags_li
                

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            # 中文没有英文wordpiece后分成几块的情况
            is_head = [1] + [0]*(len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        #  ('[CLS]', 101, 1, '[CLS]', 1),
        #  ('缘', 5357, 1, 'O', 3),
        #  ('[SEP]', 102, 1, '[SEP]', 2)
        # x为words的编码id，y为tags的编码id
        return words, x, is_heads, tags, y, seqlen


    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''pad函数的作用是在利用DataLoader生成batch数据时，由于每个batch内的各个数据长度不一致，
       因此pad函数首先求出该batch中数据最大长度，然后对该batch中长度小于最大长度的数据进行padding，
       最后通过DataLoder将数据喂入模型中。
    '''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)
    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
