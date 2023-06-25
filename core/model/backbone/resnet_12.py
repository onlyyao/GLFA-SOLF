# -*- coding: utf-8 -*-
"""
This ResNet network was designed following the practice of the following papers:
TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
"""
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.utils.dropblock import DropBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_pool = use_pool

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.use_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked),
                    1.0 - self.drop_rate,
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size ** 2
                    * feat_size ** 2
                    / (feat_size - self.block_size + 1) ** 2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

def shuffling_c(support_feat, way_num, shot_num, gen_num):

    N,c,h,w=support_feat.size() #N=n*k

    support_feat = support_feat.reshape(way_num, shot_num, c, h, w)
    support_feat = support_feat.unsqueeze(1)
    index = torch.randint(shot_num, size=[gen_num, c]).cuda() #[gen_num, c]
    one_hot = F.one_hot(index, num_classes = shot_num).permute(0,2,1).cuda() #[gen_num, c, k]——>[gen_num, k, c]
    one_hot = one_hot.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) #[1, gen_num, k, c, 1, 1]

    aug_support_feat = support_feat * one_hot #[way_num, 1, k, c, h, w] * [1, gen_num, k, c, 1, 1]
    aug_support_feat = aug_support_feat.sum(dim=2).detach() #[way_num, gen_num, c, h, w]
        
    support_feat = support_feat.permute(1,0,2,3,4,5).squeeze(0)
    support_feat = torch.cat((support_feat, aug_support_feat), dim=1) #[way_num, gen_num+k, c, h, w]
    support_feat = support_feat.reshape(way_num*(gen_num+shot_num), c, h, w)

    return support_feat

class ResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        keep_prob=1.0,
        avg_pool=True,
        drop_rate=0.0, #CIFAR-FS 要是0.0，其他是0.1
        dropblock_size=5,
        is_flatten=True,
        maxpool_last2=True,
    ):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
            use_pool=maxpool_last2,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
            use_pool=maxpool_last2,
        )
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1) # 除CIFAR以外的数据
            self.avgpool = nn.AvgPool2d(2, stride=1) #CIFAR
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.is_flatten = is_flatten
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        stride=1,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                drop_rate,
                drop_block,
                block_size,
                use_pool=use_pool,
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     if self.keep_avg_pool:
    #         x = self.avgpool(x)
    #     if self.is_flatten:
    #         x = x.view(x.size(0), -1)
    #     return x
    
    def forward(self, x, mode, way_num=5, shot_num=5, gen_num=5):
        if mode==1:
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            # for name in self.layer4.state_dict():
            #     print(name)
            # print(self.layer4.bn3.weight)
            # pdb.set_trace()
            # out4 = shuffling_c(out4, way_num, shot_num, gen_num)

        if mode==2:
            # x = shuffling_wh(x)
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)

        if mode==3:
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)

            out4 = self.avgpool(out4)
            out4 = out4.view(out4.size(0), -1)

        # if self.keep_avg_pool:
        #     out4 = self.avgpool(out4)
        # if self.is_flatten:
        #     out4 = out4.view(out4.size(0), -1)


        return out4


def resnet12(keep_prob=1.0, avg_pool=True, is_flatten=True, maxpool_last2=True, **kwargs):
    """Constructs a ResNet-12 model."""
    model = ResNet(
        BasicBlock,
        keep_prob=keep_prob,
        avg_pool=avg_pool,
        is_flatten=is_flatten,
        maxpool_last2=maxpool_last2,
        **kwargs
    )
    return model


if __name__ == "__main__":
    model = resnet12(avg_pool=True).cuda()
    data = torch.rand(10, 3, 84, 84).cuda()
    output = model(data)
    print(output.size())
