from __future__ import division      # 낮은 파이썬 버전에서도 높은 버전의 function들을 가져와 쓸 수 있게끔 하는 import

from models import *                 # models.py에서 모든 구성 class와 함수들을 가져온다.
from utils.logger import *           # utils폴더에서 logger.py파일 내부를 전부 가져온다.
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from utils.loss import compute_loss  # 같이 작성된 utils의 각 파이썬 파일에서 import를 해준다.
from test import evaluate

from terminaltables import AsciiTable # 출력하려는 부분을 table로 나타내서 출력해주는 기능

import os         # operating system(운영체제)를 제어하기 위한 library import
import sys        # 파이썬의 인터프리터를 제어하기 위한 library import
import time       # 시간을 다루기 위해 time library import
import datetime   # 현재 날짜와 시간을 받아오기 위한 libarary import
import argparse   # 인자로 받아오는 코드들을 명령어로 사용하게 해주는 library
import tqdm       # 작업진행률을 시각적으로 표시

import torch      # yolo구현을 위한 Pytorch를 가져와줍니다.
from torch.utils.data import DataLoader # 
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs") #epoch 개수 training 반복 수
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file") #definition 파일 경로(알고리즘), default로는 경로에 있는 yolov3 config파일을 가져옴
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file") #data셋 경로, default로는 경로에 있는 cocodata set을 가져옴
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model") #pretrained된 가중치가 있다면 가져와서 시작점으로 정함
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #batch를 생성하는중에 cpu스레드 개수, default=8
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension") #img 사이즈 입력, type은 int형, default는 416
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights") #긴 epcch으로 학습할 때 중간중간 모델 가중치와 바이어스 세이브하는 지점
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set") #dev셋중 평가하는 interval 지정
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training") #여러 image사이즈에 대한 training을 허용할지 말지 정함, default로는 허용
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose") #학습진행상황을 표기해주는 옵션, default로는 표시 안함
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored") #log file들이 저장되어있는 경로(directory)
    opt = parser.parse_args()
    print(opt)

    logger = Logger(opt.logdir) #logging directory 지정해서 log저장할 곳 마련

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #하나의 gpu를 사용하기 위함
    os.makedirs("output", exist_ok=True) #"output"이라는 폴더 생성
    os.makedirs("checkpoints", exist_ok=True) #"checkpoints"이라는 폴더 생성
    # exist_ok를 지정하면 폴더가 존재하지 않을 때만 생성
    # 존재하는 경우에는 아무것도 하지않음
    # Get data configuration
    data_config = parse_data_config(opt.data_config) # data_config로 받아온 data를 넣어준다.
    train_path = data_config["train"] # data_config파일에서 train부분을 뽑아낸다.
    valid_path = data_config["valid"] # data_config파일에서 valid부분을 뽑아낸다.
    class_names = load_classes(data_config["names"]) # data_config에서의 class부분의 name들을 불러오도록 한다.

    # Initiate model
    model = Darknet(opt.model_def).to(device) # 가져온 definition 파일을 가지고 가져온 models.py안의 darknet에 넣어줌으로서  darknet 객체인 model을 생성, to device
    model.apply(weights_init_normal) # darknet이 상속받은 pytorch의 nn.Module의 apply함수를 적용해서 weigts_init_normal이라는 함수를 nn.Module안의 구성된 모든층에 적용시킨다. -> 결국 모든 층에서의 parameter들을 초기화시켜주는 역할

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights)) # pretrained된 weights가 존재할 때 .pth의 확장자를 가진 파일이 존재할 때 pytorch의 load_state_dict을 이용해 모델안에 역직렬화를 통해 parameter(weight)들을 넣어준다.
        else:
            model.load_darknet_weights(opt.pretrained_weights) # .pth 확장자를 가지지 않는 다면 객체 model의 내장 함수 load_darknet_weights을 실행하여 conv층에 바로 weight와 bias를 load해준다.(

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS) # 가져온 dataset의 train_path를 여러 iamge사이즈에 대해 training하는것을 허용하면서 img_size를 받고 data_augmentation을 통해 data를 늘린다.
    dataloader = torch.utils.data.DataLoader(# pytorch의 dataloader 함수를 이용해서 dataset을 불러온다.
        dataset,
        batch_size= model.hyperparams['batch'] // model.hyperparams['subdivisions'], # 객체 model에 저장된 hyperparameter들을 불러와서 batch size를 설정해준다.
        shuffle=True, # shuffle을 통해 data를 섞는다. 
        num_workers=opt.n_cpu, # cpu 스레드에서 부가적인 process를 진행할 수 있게 했다.
        pin_memory=True, # pin_memory = True로 놓아서 data loader가 tensor를 복사하여 반환해주도록 한다.
        collate_fn=dataset.collate_fn, # 데이터 셋의 미니배치들의 list를 병합시킨다.
    ) 
   

    if (model.hyperparams['optimizer'] in [None, "adam"]): # hyperparams['optimizer']가 None이거나 adam이면 adam인것으로 간주.
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'], # optimization이 adam optimization일 때 pytorch 내부의 adam일때의 함수를 가져와 learning rate와 decay를 적용한다.
            )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])# optimization이 stochastic gradient descent optimization일 때 pytorch 내부의 SGD일때의 함수를 가져와 learning rate와 decay를 적용하고 momentum기법을 사용한다.
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).") # optimization이 adam과 sgd가 둘 다 아닐때는 optimizer를 생성하지 않는다.

    for epoch in range(opt.epochs): # 설정해준 epochs만큼 train을 하기 위함
        print("\n---- Training Model ----")
        model.train() # 모델을 train
        start_time = time.time() # 시작하는 시간 입력받아옴
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i # batches_done은 모든 epoch이 돌때 까지 iteration된 mini-batch들을 0부터 끝까지 정해주는 것!

            imgs = imgs.to(device, non_blocking=True) # cpu에 있는 image를 gpu로 옮김
            targets = targets.to(device) # image에 해당하는 label도 복사한뒤 gpu로 옮김

            outputs = model(imgs) # input인 imgs를 model에 넣어 예측값인 output를 도출한다.

            loss, loss_components = compute_loss(outputs, targets, model) # loss function을 계산해준다.

            loss.backward()  # 구한 loss function의 값으로 backward propogation을 진행해준다.

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0: # batches_done이 특정 숫자일때만 update를 시켜주도록 하기위한 if문.
                # Adapt learning rate 
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate'] # 미리 정해둔 learning rate를 가져와 lr로 저장한다.
                if batches_done < model.hyperparams['burn_in']: # burn_in의 디폴트값은 1000인데 batches_done이 burn_in보다 작을때
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in']) # learning rate를 줄여나가고 batches_done/model.hyperparams['burn_in'] 은 항상 1보다 작을때이므로 lr을 점점 작아진다.
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value  # batches_done이 1000(burn_in)보다 커질 때는 너무 작아진 lr를 사용하기 보다는 약간의 크기가 있는 lr을 사용한다.
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done) # Logger class의 객체 logger에서 scalar_summary함수에 각각의 tag, value, step을 집어넣으면 시각적으로 lr이 batched_done에 따라서 어떻게 변하는지 볼 수 있다.
                # Set learning rate
                for g in optimizer.param_groups:
                        g['lr'] = lr # 위에서 learning_rate를 변경시키고 다음에 적용할  parameters를 업데이트 시켜준다.

                # Run optimizer
                optimizer.step()  # 매개변수 갱신을 위한 optimizer의 step function 호출!
                # Reset gradients
                optimizer.zero_grad() # 누적되어서 사용되었던 gradient를 0으로 다시 만들어준다.

            # ----------------
            #   Log progress
            # ----------------
            log_str = ""
            log_str += AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])], 
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table  # 아스키 테이블로 시각적으로 요약된 정보를 얻기위해 table을 추가한다.

            if opt.verbose: print(log_str)  #verbose가 0이 아닌값을 가지면 좀 더 세부적인 정보를 가져와서 볼 수 있게 한다.

            # Tensorboard logging
            tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])), 
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done) # 시각적으로 tensorboard_log와 batches_done 사이의 관계 그래프를 보기위해 객체 logger의 함수로 부른다.
            
            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:  # 학습을 시작하고 특정한 epoch이 되면 모델을 평가하는 if문
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(  # test.py에서 불러왔던 evaluate함수 호출
                model, # 지금까지 정의했던 model을 input.
                path=valid_path, # test.py에 정의한 data_config파일에서 가져온 valid가 들어가있음.
                iou_thres=0.5, # yolov3를 돌릴때 iou의 값을 정해준다.
                conf_thres=0.1, # bounding box안에 있는 객체가 있을 확률이 0.1보다 작으면 없애도록 하는 부분.
                nms_thres=0.5, # non max suppression threshold
                img_size=opt.img_size, # 기존에 넣었던 opt.img_size를 이미지 사이즈로 넣어준다.
                batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'], # 위에서 나왔던 코드와 마찬가지로 batch_size를 정의해준다.(gpu에 mini batch를 나누어서 올리기 위함)
            )
            
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output # 위에서 구한 metrics_output이 None이 아니라면 정확한 evaluation의 지표인 precision, recall, AP를 얻기위해 각각의 변수로 받아온다.
                evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)  # metrics_output 안에 있던 precision, recall, AP, f1를 받아와 logger객체의 list_of_scalars_summary함수로 시각적으로 epoch에 대한 evalution_metrics(precision, recall, AP, f1)을 그래프로 나타낸다.
                
                if opt.verbose: # verbose는 표기하는 정보를 줄지 말지 알려주는 역할. 0이 아니라면 추가적인 정보를 주도록 함.
                    # Print class APs and mAP
                    ap_table = [["Index", "Class name", "AP"]] # ap_table이라는 배열에 index, class name, AP를 넣어 출력하기위함
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, class_names[c], "%.5f" % AP[i]]] # class들에 따라서 각가의 다른 AP를 출력을 하기위해 각각 ap_table에 넣어준다.
                    print(AsciiTable(ap_table).table) # ap_table을 아스키 테이블로 출력
                    print(f"---- mAP {AP.mean()}")   # mAP 출력             
            else:
                print( "---- mAP not measured (no detections found by model)") # 만약 metric_output이 None이라면 그에 대한 evaluation의 정보를 나타내지 않음

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
         # epoch중에 checkpoint 마다 그때의 model의 weight와 bias를 가져와 저장한다.