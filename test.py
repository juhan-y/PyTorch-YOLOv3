from __future__ import division  # 낮은 파이썬 버전에서도 높은 버전의 function들을 가져와 쓸 수 있게끔 하는 import

from models import *          # models.py에서 모든 구성 class와 함수들을 가져온다.
from utils.utils import *     # utils폴더에서 logger.py파일 내부를 전부 가져온다.
from utils.datasets import *  # utils폴더에서 datasets.py파일 내부를 전부 가져온다.
from utils.augmentations import *  # utils폴더에서 augmentations.py파일 내부를 전부 가져온다.
from utils.transforms import *     # utils폴더에서 transforms.py파일 내부를 전부 가져온다.
from utils.parse_config import *   # utils폴더에서 parse_config파일 내부를 전부 가져온다.

import os         # operating system(운영체제)를 제어하기 위한 library import
import sys        # 파이썬의 인터프리터를 제어하기 위한 library import
import time       # 시간을 다루기 위해 time library import
import datetime   # 현재 날짜와 시간을 받아오기 위한 libarary import
import argparse   # 인자로 받아오는 코드들을 명령어로 사용하게 해주는 library
import tqdm       # 작업진행률을 시각적으로 표시


import torch      # yolo구현을 위한 Pytorch를 가져와준다.
from torch.utils.data import DataLoader # data를 불러올 수 있게 기능을 가져와준다.
from torchvision import datasets  # yolo를 사용하기 위해 torchvision을 사용하고 알고리즘에 대한 기능을 가져와서 써야하므로 dataset를 가져온다.
from torchvision import transforms  # image에 대한 변환을 할 수 있도록 하는 기능 import
from torch.autograd import Variable # 자동으로 미분을 계산해줄때 필요한 기능 import
import torch.optim as optim  # optimizer에 대한 정보가 필요하므로 import

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size): # 모델에 대한 training이 완료되면 평가하기 위한 함수
    model.eval() # darknet class의 부모 class인 torch.nn.module에 있는 eval함수를 사용하여 model에 대한 평가를 진행

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS) # data를 load할 때 여러가지 설정들 적용
    dataloader = torch.utils.data.DataLoader(
        dataset,  # dataset
        batch_size=batch_size, # 입력받은 batchsize로 평가
        shuffle=False, # shuffle은 실행하지 않음
        num_workers=1, # cpu나 gpu로 올려서 진행할때 설정해주는 부분
        collate_fn=dataset.collate_fn # batch들을 묶어서 진행할 때의 설정
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor # gpu를 사용해서 계산을 할것.

    labels = [] # label들을 넣어줄 배열 선언
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        if targets is None:
            continue
            
        # Extract labels
        labels += targets[:, 1].tolist() # label을 뽑아서 추가해줌
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:]) # target에 대한 새로운 메모리를 할당하여 넣어줌
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad(): # evaluation 과정이기 때문에 gradient를 계산할 필요없음.
            outputs = to_cpu(model(imgs)) # model에 img를 넣은 결과를 cpu로 올려보냄
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres) # cpu로 올려보낸 결과를 non_max_suppression을 통해 최종 bounding box를 검출해냄.

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres) #배치에 대한 통계를 결과값에 대한 정보를 추가해준다.
    
    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None # 만약 결과가 아무것도 없다면(검출된것이 없다면) None을 return.
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] # same_metrics안의 모든 list를 풀고 합쳐서 다시 list로 만들어준다. 그 후 각각의 연결된 array를 내보낸다.
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels) # 평가 후 최종적인 지표를 도출한다.

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__": # 각각의 변수에 대한 명령어를 입력해주는 부분 자세한 명령어 설명은 train.py에서 설명한 내용이므로 생략.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu를 사용할 수 있는 환경일 때 gpu사용, 불가능한 환경이면 cpu사용.

    data_config = parse_data_config(opt.data_config) # data_config애서 parameter에 대한 정의를 가져온다.
    valid_path = data_config["valid"] # data_config안 valid에대한 정보를 가져온다.
    class_names = load_classes(data_config["names"]) # data_config파일 안 class 이름들을 가져온다.

    # Initiate model
    model = Darknet(opt.model_def).to(device) # 가져온 parameter정보로 Darknet class의 객체인 model을 만들어준다.
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path) # weight가 들어있는 경로에 .weights인 확장자를 가진 파일을 load하여 객체 model에 weight로 load해준다.
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...") 

    precision, recall, AP, f1, ap_class = evaluate( # evaluate함수에 non_max suppression에 필요한 parameter들을 넣고, model, valid path도 넣은 다음 precision, recall, AP와 같은 평가지표를 결과값으로 받는다.
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class): # mAP를 구하기 위해 AP를 가져와서 for반복문 돌림.
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
