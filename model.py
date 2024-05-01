import re
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

import torch
from transformers import BertTokenizer
from transformers import BertModel
import nltk
nltk.download('punkt')

class BERTokenizer():

    def __init__(self,opt):
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.opt = opt
    #pre-process the text data
    def text_preprocessing(self, text):

        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                    tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []
        MAX_LEN = self.opt.MAX_QUESTION_LEN
        # For every sentence...
        for sent in data:

            encoded_sent = self.tokenizer.encode_plus(
                text=self.text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                #return_tensors='pt',           # Return PyTorch tensor
                truncation=True,
                return_attention_mask=True      # Return attention mask
                )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks


# Create the question encoder base on  BertClassfier
class BertQstEncoder(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, opt,freeze_bert=True):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        @param    opt: for configuration parameter
        """
        super(BertQstEncoder, self).__init__()
        self.opt = opt
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in= self.opt.BERT_UNIT_NUM

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.bert_emb = self.bert.embeddings
        self.bert_encode_layer1 = (self.bert.encoder.layer)[0]
        self.bert_encode_layer2 = (self.bert.encoder.layer)[1]
        self.bert_encode_layer3 = (self.bert.encoder.layer)[2]
        self.bert_encode_layer4 = (self.bert.encoder.layer)[3]
        self.bert_encode_layer5 = (self.bert.encoder.layer)[4]
        self.bert_encode_layer6 = (self.bert.encoder.layer)[5]
        self.bert_encode_layer7 = (self.bert.encoder.layer)[6]
        self.bert_encode_layer8 = (self.bert.encoder.layer)[7]
        self.bert_encode_layer9 = (self.bert.encoder.layer)[8]
        self.bert_encode_layer10 = (self.bert.encoder.layer)[9]
        self.bert_encode_layer11 = (self.bert.encoder.layer)[10]
        self.bert_encode_layer12 = (self.bert.encoder.layer)[11]


    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """


        # Feed input to BERT

        with torch.no_grad():
            emb_out = self.bert_emb(input_ids, attention_mask)
            layer1= self.bert_encode_layer1(emb_out)
            layer2= self.bert_encode_layer2(layer1[0])
            layer3= self.bert_encode_layer3(layer2[0])
            layer4= self.bert_encode_layer4(layer3[0])
            layer5= self.bert_encode_layer5(layer4[0])
            layer6= self.bert_encode_layer6(layer5[0])
            layer7= self.bert_encode_layer7(layer6[0])
            layer8= self.bert_encode_layer8(layer7[0])
            layer9= self.bert_encode_layer9(layer8[0])
            layer10= self.bert_encode_layer10(layer9[0])
            layer11= self.bert_encode_layer11(layer10[0])
            layer12= self.bert_encode_layer12(layer11[0])

        word_question_representation = (layer11[0] +layer12[0])/2

        return word_question_representation


#Extract the question feature with co-attention
class QuestionFeatureExtractionAtt(nn.Module):
    '''
        Extract the question with co-attention, get from https://github.com/asdf0982/vqa-mfb.pytorch
    '''

    def __init__(self,opt):

        super(QuestionFeatureExtractionAtt, self).__init__()

        self.opt = opt
        self.NUM_QUESTION_GLIMPSE = self.opt.NUM_QUESTION_GLIMPSE

        self.JOINT_EMB_SIZE = self.opt.MFB_FACTOR_NUM * self.opt.MFB_OUT_DIM
        self.Softmax = nn.Softmax(dim=-1)

        self.Linear1_q_proj = nn.Linear(self.opt.BERT_UNIT_NUM* self.opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear2_q_proj = nn.Linear(self.opt.BERT_UNIT_NUM*self.opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)

        self.Dropout_M = nn.Dropout(p=self.opt.MFB_DROPOUT_RATIO)
        self.dropout = nn.Dropout(self.opt.BERT_DROPOUT_RATIO)
        self.Conv1_Qatt = nn.Conv2d(self.opt.BERT_UNIT_NUM, self.opt.IMAGE_CHANNEL, 1)
        self.Conv2_Qatt = nn.Conv2d(self.opt.IMAGE_CHANNEL, self.opt.NUM_QUESTION_GLIMPSE, 1)

    def forward(self,qst_encoding):

        '''
        Question Attention
        '''
        self.batch_size = qst_encoding.shape[0]
        qst_encoding = self.dropout(qst_encoding)
        qst_encoding_resh =  torch.unsqueeze(qst_encoding, 3)       # N=4 x 768 x T=14 x 1
        qatt_conv1 = self.Conv1_Qatt(qst_encoding_resh)                   # N x 512 x T x 1
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)                          # N x 2 x T x 1
        qatt_conv2 = qatt_conv2.contiguous().view(self.batch_size*2,-1)
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(self.batch_size, 2, -1, 1)
        qatt_feature_list = []
        for i in range(self.NUM_QUESTION_GLIMPSE):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)              # N x 1 x T x 1
            t_qatt_mask = t_qatt_mask * qst_encoding_resh           # N x 768 x T x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)   # N x 768 x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)       # N x 1536 x 1 x 1

        return qatt_feature_concat


#Extract the image feature with MFB and co-attention
class ImageFeatureExtractionAtt(nn.Module):

    '''
        Extract the image with co-attention, get from https://github.com/asdf0982/vqa-mfb.pytorch
    '''

    def __init__(self,opt):
        super(ImageFeatureExtractionAtt, self).__init__()
        self.opt = opt
        self.MFB_FACTOR_NUM = self.opt.MFB_FACTOR_NUM
        self.MFB_OUT_DIM = self.opt.MFB_OUT_DIM
        self.NUM_IMG_GLIMPSE =self.opt.NUM_IMG_GLIMPSE
        self.IMG_FEAT_SIZE = self.opt.IMG_FEAT_SIZE

        self.JOINT_EMB_SIZE = self.opt.MFB_FACTOR_NUM * self.opt.MFB_OUT_DIM
        self.Softmax = nn.Softmax(dim=-1)

        self.Linear1_q_proj = nn.Linear(self.opt.BERT_UNIT_NUM* self.opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear_i_proj = nn.Linear(self.opt.IMAGE_CHANNEL*self.opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Conv_i_proj = nn.Conv2d(self.opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE, 1)


        self.Dropout_M = nn.Dropout(p=self.opt.MFB_DROPOUT_RATIO)

        self.Conv1_Iatt = nn.Conv2d(self.opt.MFB_OUT_DIM, self.opt.IMAGE_CHANNEL, 1) # (1000, 512, 1)
        self.Conv2_Iatt = nn.Conv2d(self.opt.IMAGE_CHANNEL, self.NUM_IMG_GLIMPSE, 1)

    def forward(self, img_feature, qstatt_feature):

        '''
        Image Attention with MFB
        '''
        self.batch_size = img_feature.shape[0]
        q_feat_resh = torch.squeeze(qstatt_feature)                              # N x 1536
        i_feat_resh = img_feature.unsqueeze(3)                                   # N x 512 x 196 x 1
        #print(i_feat_resh.shape)
        iatt_q_proj = self.Linear1_q_proj(q_feat_resh)                                  # N x 5000
        iatt_q_resh = iatt_q_proj.view(self.batch_size, self.JOINT_EMB_SIZE, 1, 1)      # N x 5000 x 1 x 1
        iatt_i_conv = self.Conv_i_proj(i_feat_resh)                                     # N x 5000 x 196 x 1
        iatt_iq_eltwise = iatt_q_resh * iatt_i_conv
        iatt_iq_droped = self.Dropout_M(iatt_iq_eltwise)                                # N x 5000 x 196 x 1
        iatt_iq_permute1 = iatt_iq_droped.permute(0,2,1,3).contiguous()                 # N x 196 x 5000 x 1
        iatt_iq_resh = iatt_iq_permute1.view(self.batch_size, self.IMG_FEAT_SIZE, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)
        iatt_iq_sumpool = torch.sum(iatt_iq_resh, 3, keepdim=True)                      # N x 196 x 1000 x 1
        iatt_iq_permute2 = iatt_iq_sumpool.permute(0,2,1,3)                             # N x 1000 x 196 x 1
        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_permute2)) - torch.sqrt(F.relu(-iatt_iq_permute2))
        iatt_iq_sqrt = iatt_iq_sqrt.reshape(self.batch_size, -1)                           # N x 196000
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(self.batch_size, self.MFB_OUT_DIM, self.IMG_FEAT_SIZE, 1)  # N x 1000 x 196 x 1

        iatt_conv1 = self.Conv1_Iatt(iatt_iq_l2)                    # N x 512 x 196 x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.Conv2_Iatt(iatt_relu)                     # N x 2 x 196 x 1
        iatt_conv2 = iatt_conv2.view(self.batch_size*self.NUM_IMG_GLIMPSE, -1)
        iatt_softmax = self.Softmax(iatt_conv2)
        iatt_softmax = iatt_softmax.view(self.batch_size, self.NUM_IMG_GLIMPSE, -1, 1)
        iatt_feature_list = []
        for i in range(self.NUM_IMG_GLIMPSE):
            t_iatt_mask = iatt_softmax.narrow(1, i, 1)              # N x 1 x 196 x 1
            t_iatt_mask = t_iatt_mask * i_feat_resh                 # N x 512 x 196 x 1
            t_iatt_mask = torch.sum(t_iatt_mask, 2, keepdim=True)   # N x 512 x 1 x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.cat(iatt_feature_list, 1)       # N x 1024 x 1 x 1
        iatt_feature_concat = torch.squeeze(iatt_feature_concat)    # N x 1024
        return iatt_feature_concat