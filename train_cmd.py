#-*-coding:utf-8-*-

cmds = [
    "python main.py -d /media/hkuit155/NewDisk/imagenet -dset imagenet -a resdg50 -lr 0.05 --weight-decay 4e-5 --epochs 200 --gpu-id 0 --den-target 0.5 --pretrained pytorch --lr-mode cosine --batch-size 256 --checkpoint imagenet_resnet50/DGNet_50",
    "python main.py -d /media/hkuit155/NewDisk/imagenet -dset imagenet -a resdg50 -lr 0.05 --weight-decay 4e-5 --epochs 200 --gpu-id 0 --den-target 0.5 --pretrained pytorch --lr-mode cosine --batch-size 256 --checkpoint imagenet_resnet50/DPACS_50",
    "python main.py -d /media/hkuit155/NewDisk/imagenet -dset imagenet -a resdg50 -lr 0.05 --weight-decay 4e-5 --epochs 200 --gpu-id 0 --den-target 0.5 --pretrained pytorch --lr-mode cosine --batch-size 256 --checkpoint imagenet_resnet50/DPACS_full_50",

    "python main.py -d /media/hkuit155/NewDisk/imagenet -dset imagenet -a mobilenet_v2_dg -lr 0.05 --weight-decay 4e-5 --epochs 200 --gpu-id 0 --den-target 0.5 --pretrained pytorch --lr-mode cosine --batch-size 256 --checkpoint imagenet_mobile/DGNet_50",
    "python main.py -d /media/hkuit155/NewDisk/imagenet -dset imagenet -a mobilenet_v2_dg -lr 0.05 --weight-decay 4e-5 --epochs 200 --gpu-id 0 --den-target 0.5 --pretrained pytorch --lr-mode cosine --batch-size 256 --checkpoint imagenet_mobile/DPACS_50",
    "python main.py -d /media/hkuit155/NewDisk/imagenet -dset imagenet -a mobilenet_v2_dg -lr 0.05 --weight-decay 4e-5 --epochs 200 --gpu-id 0 --den-target 0.5 --pretrained pytorch --lr-mode cosine --batch-size 256 --checkpoint imagenet_mobile/DPACS_full_50",
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
