## kookmin university '차량지능기초' first assignment.
## Hi I'm juhan yoon. there's summary below the lines. please read the lines before see my sources.

# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

## There's a lot of codes in this original github, so i couldn't write explanation of all files(python file, config, etc..).

## So you just watch codes in train, test, models python file. there are explanation I wrote myself.

## Explanation of this project
I used PyTorch-YOLOv3 to implement computer vision for self-driving in deep-learning.( using google colaboratory)
The original doucumnet recommended using COCO dataset, but I don't want to use typical dataset.
so I used vehicle-openimages dataset in roboflow. you can find out what it is in https://public.roboflow.com/object-detection/vehicles-openimages
This dataset doesn't have many images, there're 627 images including test, train, dev set.
So I trained my model more than usual. please keep in your mind and see my model.

## Address of colab and original document I forked from.
https://colab.research.google.com/drive/1tFRKda4d6vKD6geag0WNzkU8y5DA8X9U?usp=sharing For creating model, I wrote codes on Colab.
https://github.com/eriklindernoren/PyTorch-YOLOv3 erik lindernoren ML engineer at Apple.


![image](https://user-images.githubusercontent.com/81463668/113806071-a0005200-979c-11eb-9119-ea0836336c90.png)

train.py는 간략하게 말하자면 제작자가 미리 만든 hyperparameter들을 이용해 모델에 적용,
그리고 forward, backward를 진행하는 파일이다. 파일 이름대로 전체적인 training을 할 수 있도록 만드는 python 파일이다. 실제 training을 진행할 때 이 train.py를 이용하여 model을 학습 시킬 것이다.

![image](https://user-images.githubusercontent.com/81463668/113806095-aabae700-979c-11eb-911e-f744c4d01b3a.png)
![image](https://user-images.githubusercontent.com/81463668/113806100-adb5d780-979c-11eb-9ba8-cdf392900c0b.png)
![image](https://user-images.githubusercontent.com/81463668/113806105-b1495e80-979c-11eb-85d9-9d6d6d733be7.png)
![image](https://user-images.githubusercontent.com/81463668/113806110-b3abb880-979c-11eb-8cd7-84d9f7a82263.png)
![image](https://user-images.githubusercontent.com/81463668/113806112-b60e1280-979c-11eb-9c0d-a59ccbdb9ef8.png)
![image](https://user-images.githubusercontent.com/81463668/113806117-b7d7d600-979c-11eb-9fcc-2715864bed9a.png)
![image](https://user-images.githubusercontent.com/81463668/113806124-ba3a3000-979c-11eb-92da-1518b8c42242.png)
![image](https://user-images.githubusercontent.com/81463668/113806126-bc03f380-979c-11eb-9bca-5a38cb5c721d.png)


![image](https://user-images.githubusercontent.com/81463668/113806155-c9b97900-979c-11eb-8887-359004ae83b4.png)

test.py는 training된 model을 평가하고(evaluate) 평가를 통해 얻은 정량적인 지표를 얻을 수 있도록 수학적으로 AP나 precision, recall등을 계산해주는 함수들을 이용한다.
이는 이후에 tensorboard를 이용하여 시각적으로 그래프화 시킬 것이다. 

![image](https://user-images.githubusercontent.com/81463668/113806174-d1791d80-979c-11eb-8ac0-5b5c8cd3576d.png)
![image](https://user-images.githubusercontent.com/81463668/113806182-d3db7780-979c-11eb-9d77-c5c20c0f8d04.png)
![image](https://user-images.githubusercontent.com/81463668/113806184-d5a53b00-979c-11eb-8032-530331e81880.png)
![image](https://user-images.githubusercontent.com/81463668/113806190-d6d66800-979c-11eb-8645-600844dc3cf7.png)
![image](https://user-images.githubusercontent.com/81463668/113806193-d8079500-979c-11eb-9ce5-6a25889b22cf.png)


![image](https://user-images.githubusercontent.com/81463668/113806199-da69ef00-979c-11eb-9df7-069d4182a1bb.png)

models.py는 여러 함수로 이루어져있는데 대략적으로 말하자면 model을 구성하기 위한 층들을 모듈화시켜서 사용하기 용이하게 하고 darknet, yololayer 같은 class를 정의해서 기존에 이미 만들었던 config파일안의 parameter나 층에 대한 정보들을 가져와서 YOLO알고리즘을 구현한다. 추가적으로 training할 때 weight를 저장하여 training중 오류가 나더라도 최근에 저장된 weight를 불러와서 다시 학습시키거나 또는 이런 저장된 weight로 전이학습(transfer learning)도 가능하게 한다.

![image](https://user-images.githubusercontent.com/81463668/113806223-e2299380-979c-11eb-95ce-27f32636bb1a.png)
![image](https://user-images.githubusercontent.com/81463668/113806231-e5bd1a80-979c-11eb-9b6a-8bb37da26482.png)
![image](https://user-images.githubusercontent.com/81463668/113806241-e950a180-979c-11eb-9c33-573160332276.png)
![image](https://user-images.githubusercontent.com/81463668/113806242-eb1a6500-979c-11eb-88e5-2cae0dd046da.png)
![image](https://user-images.githubusercontent.com/81463668/113806245-ece42880-979c-11eb-980f-78bcfab94918.png)
![image](https://user-images.githubusercontent.com/81463668/113806251-eeadec00-979c-11eb-9af2-1e3715304258.png)
![image](https://user-images.githubusercontent.com/81463668/113806257-f077af80-979c-11eb-8680-fbbc53ba715d.png)
![image](https://user-images.githubusercontent.com/81463668/113806264-f2417300-979c-11eb-82b7-43bc97b89f02.png)
![image](https://user-images.githubusercontent.com/81463668/113806268-f40b3680-979c-11eb-8337-e62b117dc85f.png)
![image](https://user-images.githubusercontent.com/81463668/113806271-f5d4fa00-979c-11eb-99cb-0e22af919772.png)
![image](https://user-images.githubusercontent.com/81463668/113806277-f79ebd80-979c-11eb-84df-01da44510bac.png)



## ※ 파일 구성

![image](https://user-images.githubusercontent.com/81463668/113806286-fb324480-979c-11eb-97fd-2001b65da7dd.png)


- assets 폴더 안에는 4장의 사진이 있는데 yolo 알고리즘을 테스트할 목적으로 있는 것으로 판단된다. 이 프로젝트를 진행하는데는 중요한 부분은 아니다.
- config폴더 안에는
![image](https://user-images.githubusercontent.com/81463668/113806316-09806080-979d-11eb-8e3c-bc5a59c02f9c.png)

이런식으로 config와 sh, data파일이 들어있는데 coco.data 파일은 COCO dataset과 관련된 정보를 담은 파일이므로 이 또한 이번 프로젝트에서 사용되지 않은 파일이다.
이 대신 3번째에 있는 custom.data라는 파일이 내가 이번 프로젝트에 사용한 roboflow에서 가져온 dataset에 대한 정보가 들어있는 파일이다.

![image](https://user-images.githubusercontent.com/81463668/113806323-0f764180-979d-11eb-9922-5bb6ab8fcbd8.png)

여기서 classes는 class의 개수를 의미한다. 내가 가져온 데이터셋의 class는 차, 버스, 오토바이, 트럭, 앰뷸런스로 이루어져있다.
train, valid가 의미하는 것은 각 경로에서 txt확장자를 가진 파일인 label파일을 가져온다는 말이다. names는 class의 이름을 가져오는 것이다.

create_custom_model.sh파일은 yolo-custom config파일과 비슷한 구성을 가지고 있는데 원할시에는 층의 정보나 hyperparameter를 건드려서 바꿀 수 있다. 이렇게 바꾼 내용으로 돌리는 것도 가능하다. 하지만 이번 프로젝트에서는 yolo알고리즘을 정석적으로 사용하여 내가 custom한 데이터셋에 적용시키는 것을 우선으로 두었다.

그 이외에 여러 가지 다른 config파일들이 있는데 yolo3.cfg파일은 coco dataset을 돌리려고 할 때 작성자가 만들어둔 정석적인 config파일이고 yolo3-tiny.cfg파일은 층의 개수가 좀 더 적은 yolo알고리즘 config파일이라고 보면 되겠다.

![image](https://user-images.githubusercontent.com/81463668/113806345-14d38c00-979d-11eb-8cbc-363090326620.png)

이 checkpoints라는 폴더는 training이후에 자동적으로 생기는데 이 폴더안에는 model을 epoch단위로 training시킬 때 1epoch마다 weight나 다른 parameter들을 저장하는 부분이다.

![image](https://user-images.githubusercontent.com/81463668/113806358-1a30d680-979d-11eb-911c-2246bc251c7c.png)

(이런식으로 pth파일이 생성되어있다.)


![image](https://user-images.githubusercontent.com/81463668/113806374-2026b780-979d-11eb-8356-06a4b216927f.png)

다음으로는 data폴더인데 data폴더 내부에는 images와 labels는 roboflow에서 다운받은 dataset을 넣어주었다. classes.names는 class이름을 저장한 파일이다.
train.txt와 valid.txt는 train과 valid로 나눈 label들을 불러올 때 쓰는 파일이다.
내부에는 각 label의 경로가 들어있다.

![image](https://user-images.githubusercontent.com/81463668/113806384-2452d500-979d-11eb-9210-8bbd1f0e4795.png)

train을 시킬때마다 log가 생겨서 logs폴더에 들어가게된다. 
내부에는 각종 parameter와 계산했던 평균과 분산, AP의 값들이 들어있다.
이를 가져다가 tensorboard에서 비교할 수도 있다.

output폴더에는 아무것도 들어가있지않다. 아마도 코드를 만든 개발자들이 coco dataset으로 돌릴때에 넣으려고한 폴더같다.

![image](https://user-images.githubusercontent.com/81463668/113806395-2ae14c80-979d-11eb-8325-3783d7958366.png)

utils폴더에는 우리가 train.py나 test.py를 통해 model을 train시키고 평가할 때 쓰이는 함수들이 정의되어있는 python파일들이 저장되어있다. init.py에는 현재는 아무 함수가 정의되어 있지않다. augmentations.py에는 이미지들을 augmentation기법으로 데이터개수를 늘려줄 수 있는 함수가 들어가있다. 하지만 이번 프로젝트에서는 사용되지 않았다.
datasets.py은 dataset인 image파일들을 padding하고 resize하는 함수들이 들어있으며, 새로운 타입으로 변환시키는 함수도 들어가있다.
logger.py는 좀 전에 봤었던 logs폴더안에 들어가는 log들을 만들어줄 수 있게하는 함수들이 들어가있다.
loss.py는 파일명대로 loss function을 계산해주는 함수가 들어가있다.
parse_config.py는 개발자가 정해놓은 hyper parameter들을 parser형태로 명령어로 만들어서 입력할 수 있게끔하는 함수들이 들어있다.
transforms.py에는 augmentation.py에 있었던 augmentation 기법을 실제 사용하여 클래스를 정의하고 라벨, tensor, image들의 상태를 변환시키거나 사이즈를 변환시키는 class들이 들어있다.
utils.py에는 weight를 초기화하고 label의 class를 load하고 bounding box들을 rescale하고, ap 계산, bounding iou를 계산해주는 함수들이 들어있다. 추가적으로 yolo기법에서 중요한
non max suppression 함수도 들어가있다.

![image](https://user-images.githubusercontent.com/81463668/113806402-2e74d380-979d-11eb-9b11-5341f1c99da9.png)

weights폴더안에는 말 그대로 가중치들이 들어있는 폴더이다.
darknet53.conv.74는 이번 model의 module을 구성하는 convolutional layer에 대한 정보들이 들어가있다. download_weights.sh는 pretrained된 가중치로 model의 가중치를 초기화한 후에 적용시켜서 training을 시작한다. yolov3-tiny.weights와 yolov3.weights는 darknet으로 pretrained된 가중치이다. 이번 코드실행에서는 사용되지 않는다.

![image](https://user-images.githubusercontent.com/81463668/113806415-346ab480-979d-11eb-8754-e7ddee4055f5.png)

License파일은 말그대로 License에 관한 정보가 들어있다.
README.md는 이 코드를 만든 개발자가 어떻게 코드가 진행되는지 부가적인 설명도 들어있다. detect.py는 최종적으로 개발자들이 만든 python파일과 config파일들을 이용해 model을 구성하고 training이후에 평가하는 부분도 있다.
requirements.txt는 model구성에 필요한 여러 가지 library(numpy, torch, torchvision...)들을 import한다. 












