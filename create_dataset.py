import os
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io, transform
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from vgg import VGG19
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch

import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

transform = transforms.Compose([transforms.Pad((0, 85), fill=0, padding_mode='constant'),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])

QVQA_PATH = "/content/drive/MyDrive/OVQA_publish/"


def compute_prerequisites():
    ANS_LABLE_DICT = {}
    Q_TYPE_LABLE_DICT = {}
    ANS_TYPE_LABLE_DICT = {}
    IMG_ORGAN_LABLE_DICT = {}
    with open(QVQA_PATH + "valset.json") as f_val:
        data_val = json.load(f_val)

    with open(QVQA_PATH+ "trainset.json") as f_train:
        data_train = json.load(f_train)

    with open(QVQA_PATH + "testset.json") as f_test:
        data_test = json.load(f_test)

    data = data_val + data_train + data_test
    i = 0
    j = 0
    k = 0
    l = 0
    for elem in data:
        if elem["answer"] not in ANS_LABLE_DICT.keys():
            # remove special characters and make it all lower
            ANS_LABLE_DICT[elem["answer"]] = i
            i += 1

        if elem["question_type"] not in Q_TYPE_LABLE_DICT.keys():
            Q_TYPE_LABLE_DICT[elem["question_type"]] = j
            j += 1

        if elem["answer_type"] not in ANS_TYPE_LABLE_DICT.keys():
            ANS_TYPE_LABLE_DICT[elem["answer_type"]] = k
            k += 1

        if elem["image_organ"] not in IMG_ORGAN_LABLE_DICT.keys():
            IMG_ORGAN_LABLE_DICT[elem["image_organ"]] = l
            l += 1

    return ANS_LABLE_DICT, Q_TYPE_LABLE_DICT, ANS_TYPE_LABLE_DICT

ANS_LABLE_DICT, Q_TYPE_LABLE_DICT, ANS_TYPE_LABLE_DICT = compute_prerequisites()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19_model = VGG19().to(device)

class OVQADataset(Dataset):
    """OVQA images and questions dataset."""

    def __init__(self, json_file, root_dir, task, phase, transform=None):
        """
        Arguments:
            json_file (string): Path to the json file with questions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_file) as f:
          self.question_data = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.task = task
        self.phase = phase

    def __len__(self):
        return len(self.question_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.question_data[idx]["image_name"])

        image = Image.open(img_name).convert('RGB')
        if self.transform:
          image = self.transform(image)
          #print(image.shape)

        question = self.question_data[idx]["question"]
        answer =  self.question_data[idx]["answer"]
        question_type = self.question_data[idx]["question_type"]
        answer_type =  self.question_data[idx]["answer_type"]
        image_organ = self.question_data[idx]["image_organ"]
        qid = self.question_data[idx]["qid"]

        #print(image.size())
        sample = {'image': vgg19_model(image[None,...].to(device)),
            'question': question,
            'answer_label':F.one_hot(torch.tensor([[ANS_LABLE_DICT[answer]]]), len(ANS_LABLE_DICT)),
            'question_type_label':F.one_hot(torch.tensor([[Q_TYPE_LABLE_DICT[question_type]]]), len(Q_TYPE_LABLE_DICT)),
            'answer_text': answer,
            'answer_type_label':F.one_hot(torch.tensor([[ANS_TYPE_LABLE_DICT[answer_type]]]), len(ANS_TYPE_LABLE_DICT)),
            'qid': qid}
            #'image_organ_label': F.one_hot(torch.tensor([[IMG_ORGAN_LABLE_DICT[image_organ]]]), len(IMG_ORGAN_LABLE_DICT))}

        x = (sample['image'], sample['question'])
        if self.task == "main_answer":
            y = sample['answer_label']
        elif self.task == "question_classification":
            y = sample["question_type_label"]
        else:
            assert False

        return x, y
    

def ovqa_dataloader(batch_size, num_workers, size=228):
    '''
        Load our dataset with dataloader for the train and valid data
    '''

    tasks = ["main_answer", "question_classification"]
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        data_loader[d] = {}
        iter_data_loader[d] = {}
        for phase in ['train', 'val', 'test']:
            shuffle = True if phase == 'train' else False
            drop_last = True if phase == 'train' else False
            ovqa_dataset = OVQADataset(json_file=f'{QVQA_PATH}{phase}set.json',
                                    root_dir=f'{QVQA_PATH}img/', task=d, phase=phase, transform=transform)
            
            data_loader[d][phase] = DataLoader(ovqa_dataset, 
                                              num_workers=num_workers, 
                                              pin_memory=False,  # Maybe False
                                              batch_size=batch_size, 
                                              shuffle=shuffle,
                                              drop_last=drop_last)
            iter_data_loader[d][phase] = iter(data_loader[d][phase])
    return data_loader, iter_data_loader