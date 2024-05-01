import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
from LibMTL.trainer import Trainer
from create_dataset import ovqa_dataloader, ANS_LABLE_DICT, Q_TYPE_LABLE_DICT, ANS_TYPE_LABLE_DICT
from model import BERTokenizer, BertQstEncoder, ImageFeatureExtractionAtt, QuestionFeatureExtractionAtt

from LibMTL.utils import set_random_seed, set_device

import argparse

def parse_args(parser):
    parser.add_argument('--dataset', default='office-31', type=str, help='office-31, office-home')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

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


class VQAClassifierModel(nn.Module):
    '''
        Fusion with MFB,  get from https://github.com/asdf0982/vqa-mfb.pytorch
    '''

    def __init__(self, opt):
        super(VQAClassifierModel, self).__init__()
        self.opt = opt

        self.JOINT_EMB_SIZE = self.opt.MFB_FACTOR_NUM * self.opt.MFB_OUT_DIM

        self.MFB_OUT_DIM = self.opt.MFB_OUT_DIM
        self.MFB_FACTOR_NUM = self.opt.MFB_FACTOR_NUM
        NUM_OUTPUT_UNITS = self.opt.NUM_OUTPUT_UNITS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BERTokenizer(self.opt)
        self.bert_model = BertQstEncoder(self.opt)

        self.qst_feature_att = QuestionFeatureExtractionAtt(self.opt)
        self.img_feature_att = ImageFeatureExtractionAtt(self.opt)

        self.Linear2_q_proj = nn.Linear(self.opt.BERT_UNIT_NUM*self.opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear_i_proj = nn.Linear(self.opt.IMAGE_CHANNEL*self.opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)

        self.Dropout_M = nn.Dropout(p=self.opt.MFB_DROPOUT_RATIO)

        # self.Linear_predict_1 = nn.Linear(self.opt.MFB_OUT_DIM, NUM_OUTPUT_UNITS)
        # self.Linear_predict_2 = nn.Linear(self.opt.MFB_OUT_DIM, len(Q_TYPE_LABLE_DICT))
        # self.Linear_predict_3 = nn.Linear(self.opt.MFB_OUT_DIM, len(ANS_TYPE_LABLE_DICT))


    def forward(self, x):
        img, qst = x
        self.batch_size = img.shape[0]
        image_feature = img
        input_ids, attention_mask = self.tokenizer.preprocessing_for_bert(qst)
        question_feature = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))
        question_feature = question_feature.transpose(1, 2)      # N=4 x 768 x T=14

        q_featatt = self.qst_feature_att(question_feature)      # N x 1536

        iatt_feature_concat = self.img_feature_att(image_feature,q_featatt)          # N x 1024

        '''
        Fine-grained Image-Question MFB fusion
        '''
        q_feat_resh = torch.squeeze(q_featatt)
        mfb_q_proj = self.Linear2_q_proj(q_feat_resh)               # N x 5000
        mfb_i_proj = self.Linear_i_proj(iatt_feature_concat)        # N x 5000
        mfb_iq_eltwise = torch.mul(mfb_q_proj, mfb_i_proj)          # N x 5000
        mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)
        mfb_iq_resh = mfb_iq_drop.view(self.batch_size, 1, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)

        return mfb_l2


def main(params):

    kwargs, optim_param, scheduler_param = prepare_args(params)

    task_name = ["main_answer", "question_classification"]
    class_no = {"main_answer": len(ANS_LABLE_DICT), "question_classification": len(Q_TYPE_LABLE_DICT)}
    
    task_dict = {task: {'metrics': ['Acc'],
                    #TODO: Add our metrics (TOP_K, BLEU_Score etc.)
                    'metrics_fn': AccMetric(),
                    'loss_fn': CELoss(),
                    'weight': [1]} for task in task_name}
    
    data_loader, _ = ovqa_dataloader(32, 0) # (32, 0)
    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}

    # encoder = VQAClassifierModel(opt=opt)
    decoders = nn.ModuleDict({task: nn.Sequential(nn.Linear(opt.MFB_OUT_DIM, class_no[task]),
                                                  nn.Softmax(dim=1)) for task in list(task_dict.keys())})

    ovqa_model = Trainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=VQAClassifierModel, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        ovqa_model.train(train_dataloaders=train_dataloaders, 
                          val_dataloaders=val_dataloaders,
                          test_dataloaders=test_dataloaders, 
                          epochs=params.epochs)
    elif params.mode == 'test':
        ovqa_model.test(test_dataloaders)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)