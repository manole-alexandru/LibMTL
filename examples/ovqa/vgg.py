import torchvision
import torch
import torch.nn as nn
#Extract image feature

class VGG19(nn.Module):
    def __init__(self):
        '''
             We remove all the fully-connected layers in the VGG19 network and the convolution outputs of different feature scales
                are concatenated after global average pooling and l2-norm to form a 1984-dimensional vector to represent the image
        '''
        super(VGG19,self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)

        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9])
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])
        self.Conv6 = nn.Sequential(*list(vgg_model.features.children())[30:36])

        self.avgpool = nn.Sequential(list(vgg_model.children())[1])

    def forward(self,image):

        with torch.no_grad():
            out1 = self.Conv1(image)
            out2 = self.Conv2(out1)
            out3 = self.Conv3(out2)
            out4 = self.Conv4(out3)
            out5 = self.Conv5(out4)          # [N, 512, 14, 14]
            out6 = self.Conv6(out5)

            out7 = self.avgpool(out6)

        #global average pooling
        out1 = out1.mean([2,3],keepdim=True)
        out2 = out2.mean([2,3],keepdim=True)
        out3 = out3.mean([2,3],keepdim=True)
        out4 = out4.mean([2,3],keepdim=True)
        out5 = out5.mean([2,3],keepdim=True)
        out6 = out6.mean([2,3],keepdim=True)
        out7 = out7.mean([2,3],keepdim=True)


        concat_features = torch.cat([out1, out2, out3, out4,out5], 1)

        #l2-normalized feature vector
        l2_norm = concat_features.norm(p=2, dim=1, keepdim=True).detach()
        concat_features = concat_features.div(l2_norm)               # l2-normalized feature vector


        batch_size = concat_features.shape[0]
        embedding_dim_size = concat_features.shape[1]
        image_feature = concat_features.view(batch_size, embedding_dim_size, -1).squeeze(0) # [N, 1984, 1]


        return image_feature