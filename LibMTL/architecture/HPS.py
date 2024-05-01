import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--SEED', type=int, default=97)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=64)
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=1067)
    parser.add_argument('--MAX_QUESTION_LEN', type=int, default=17)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=1472)
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=1e-4)
    parser.add_argument('--LAMNDA', type=float, default=0.0001)
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--BERT_UNIT_NUM', type=int, default=768)
    parser.add_argument('--BERT_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=1)
    parser.add_argument('--IMG_INPUT_SIZE', type=int, default=224)
    parser.add_argument('--NUM_EPOCHS', type=int, default=200)
    args = parser.parse_args(args=[])
    return args

opt = parse_opt()

class HPS(AbsArchitecture):
    r"""Hard Parameter Sharing (HPS).

    This method is proposed in `Multitask Learning: A Knowledge-Based Source of Inductive Bias (ICML 1993) <https://dl.acm.org/doi/10.5555/3091529.3091535>`_ \
    and implemented by us. 
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(HPS, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class(opt=opt)
