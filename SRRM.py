import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from resnet_cbam import resnet50_cbam

class SemBranch_9(nn.Module):
    """
    Resnet50 + ChAM
    """

    def __init__(self, scene_classes, semantic_classes=152):
        super(SemBranch_9, self).__init__()
        base = resnet50_cbam()


        # Semantic Branch
        self.in_block_sem = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(semantic_classes, 256, kernel_size=7, stride=2,  padding=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.in_block_sem_1 = base.layer2
        self.in_block_sem_2 = base.layer3
        self.in_block_sem_3 = base.layer4

        # Semantic Scene Classification Layers
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_SEM = nn.Linear(2048, scene_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, sem):
        # Semantic Branch
        y = self.in_block_sem(sem)

        y2 = self.in_block_sem_1(y)
        y3 = self.in_block_sem_2(y2)

        y4 = self.in_block_sem_3(y3)

        # Semantic Classification Layer
        act_sem = self.avgpool(y4)
        act_sem = act_sem.view(act_sem.size(0), -1)

        act_sem = self.dropout(act_sem)
        act_sem = self.fc_SEM(act_sem)
        fea_map = y

        return act_sem, fea_map

    def loss(self, x, target):
        # Check inputs
        assert (x.shape[0] == target.shape[0])

        # Classification loss
        loss = self.criterion(x, target.long())

        return loss
