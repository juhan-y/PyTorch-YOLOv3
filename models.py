from __future__ import division # 낮은 파이썬 버전에서도 높은 버전의 function들을 가져와 쓸 수 있게끔 하는 import
from itertools import chain  # list들을 연결하기 위한 import 

import torch     # NN을 구성하기 위한 pytorch를 가져와 준다.
import torch.nn as nn
import torch.nn.functional as F # 컨볼루션 연산을 위한 conv층을 가져오기 위해 nn.functional을 가져오고 F로 간단하게 쓸 수 있게끔 한다.
from torch.autograd import Variable # 자동으로 미분연산을 진행해주기 위해서 import
import numpy as np  # 기본적인 연산을 위해서 numpy import!

from utils.parse_config import *  # 코드작성자의 utils폴더 내에 있는 parse_config.py를 가져온다.
from utils.utils import build_targets, to_cpu, non_max_suppression  # utils폴더 안에 있는 utils.py 파일 내의 함수들을 import한다.

import matplotlib.pyplot as plt
import matplotlib.patches as patches  # matplot 라이브러리를 가져와 이미지나 그래프를 그리기 위함


def create_modules(module_defs): # config파일에 의해 정의된 module_defs로 module을 생성하는 함수. config
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # 이 모델에서 쓰는 알고리즘(yolo)의 module에 대한 첫번째 내용을 갖고와준다.
    hyperparams.update({
        'batch': int(hyperparams['batch']),  # 갖고오는 module의 첫번째 원소들 중 batch를 넣어준다.(갱신)
        'subdivisions': int(hyperparams['subdivisions']), # gpu에 모두 한꺼번에 올려서 학습하기 힘들기 때문에 subdivisions(int)로 나누어 올리게 되는데 이를 module_defs에서 가져와준다.
        'width': int(hyperparams['width']), # 넣어주는 input에 대한 width 정보
        'height': int(hyperparams['height']), # 넣어주는 input에 대한 height 정보
        'channels': int(hyperparams['channels']), # 넣어주는 input에 대한 channels 정보
        'optimizer': hyperparams.get('optimizer'), # optimizer를 어떤 것을 쓸지에 관한 정보를 가져온다.
        'momentum': float(hyperparams['momentum']), # 모멘텀 기법을 사용하기 위해 내부변수에 관한 정보를 가져온다.
        'decay': float(hyperparams['decay']), # learning rate를 decay하기 위해 감쇠율을 가져온다.
        'learning_rate': float(hyperparams['learning_rate']), # 학습할 때 필요한 learning_rate를 가져온다.
        'burn_in': int(hyperparams['burn_in']), # batches_done이랑 비교하기 위한 int값, 1000
        'max_batches': int(hyperparams['max_batches']),  
        'policy': hyperparams['policy'], # steps(40만, 45만)의 값이 들어있는데 train.py에서 batches_done이 많은 차례가 지나서 값이 커지게 되면 learning rate를 더 작게 만들어 더 정확한 최적화지점에 도달하게 하기 위함
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")), 
                             map(float, hyperparams["scales"].split(","))))
    }) # 왜 policy와 같은 내용이 lr_steps에도 들어있는지는 모르겠지만 policy는 실제 사용하지는 않고 이 lr_steps라는 값을 쓴다. batches_done이 큰값을 가질 때 learning rate값을 감소시켜준다.
    assert hyperparams["height"] == hyperparams["width"], # 데이터셋을 불러올때 이미지의 width와 height가 같은 지은 지 확인해준다.
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]] # input 이미지의 channel수를 가져와 넣어준다.
    module_list = nn.ModuleList() # 리스트안에 있는 sub module들을 불러와준다.
    for module_i, module_def in enumerate(module_defs): # module_defs에는 module을 구성하는 net들을 가져와서 for반복문을 돌린다.
        modules = nn.Sequential() # module을 구성하기 위해 편하게 쓸 수 있는 nn의 Sequential 함수를 가져온다.

        if module_def["type"] == "convolutional": # module_def에서 가져온 net의 type이 convolutional 일때
            bn = int(module_def["batch_normalize"]) # batch normalization을 가져와서 변수져와서 변수에 저장
            filters = int(module_def["filters"]) # filter의 개수를 가져온다.
            kernel_size = int(module_def["size"]) # filters의 width(height)를 가져온다.
            pad = (kernel_size - 1) // 2 # padding의 크기를 임의로 계산하여 넣어준다.
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,  # 위에서 가져온 구성요소로 convolutional 층을 만들어준다. 각각의 batch_norm이나 padding, stride, 필터의 정보를 넣어줌. 이렇게 넣어준 conv층을 module에 추가해준다.
                ),
            )
            if bn: # bn이 존재하면 batchnormalization 층을 추가해줌
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky": # 마지막으로 활성화함수로 leakyReLu function을 적용
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool": # net이 maxpool이면 max pooling층 추가 (내부적으로 조건에따라 zero padding층이 추가되기도 함.)
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample": # 말그대로 upsampling을 하기위해서 차원을 늘려주는 것.
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample) # 모듈에 추가

        elif module_def["type"] == "route": # residaual net을 쓸때 얼마나 전 층의 feature map을 가져올 것인지 정함
            layers = [int(x) for x in module_def["layers"].split(",")] # ','로 나누어 for문에 넣어준다.
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", nn.Sequential()) # 모듈에 추가

        elif module_def["type"] == "shortcut": # resnet에 있는 residual net 추가
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential()) # 모듈에 추가

        elif module_def["type"] == "yolo": # yolo layer 추가
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"]) # module_def에서 classes관한 내용을 가져옴
            img_size = int(hyperparams["height"]) # module_def에서 height관한 내용을 가져옴
            ignore_thres = float(module_def["ignore_thresh"]) # bouding box 무시하는 threshold를 가져와 정의
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size, ignore_thres)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules) # 정의한 모듈을 module list에 추가
        output_filters.append(filters)  # 여기서 사용한 filter의 개수를 가져와 넣어준다.

    return hyperparams, module_list


class Upsample(nn.Module): # nn.Module안에 있는 층들을 upsampling하는 것.
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()# 부모생성자인 nn.Module의 생성자를 호출
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode) # 결측값에 대한 보간법 사용
        return x

class YOLOLayer(nn.Module): # nn.Module을 부모로하는 YOLO층의 class
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_size, ignore_thres): # yolo 알고리즘을 돌리기위해 anchor박스에 대한 정보나 class에 대한 정보를 가져온다.
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors) # 앵커박스 개수
        self.num_classes = num_classes # classes 가져오기
        self.ignore_thres = 0.5 # bounding box없애는 기준을 정해준다.
        self.mse_loss = nn.MSELoss() # 손실함수에 관한 버전 가져온다.
        self.bce_loss = nn.BCELoss() # binary classification 손실함수 가져온다. 
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1) # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2) # anchors list를 unpacking한 후에 (-1,2)라는 shape로 tensor의 크기를 바꾸어준다.
        self.register_buffer('anchors', anchors) # gpu연산을 가능케 함
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.img_size = img_size # 객체생성시에 가져오는 img_size로 정의한다.
        self.stride = None # stride는 정의하지 않음

    def forward(self, x):  # bounding box를 만들어주는 함수
        stride = self.img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # x에 대한 tensor를 permute(transpose와 비슷한 기능) 하고 permute가 진행되기 위해서 contiguous한 tensor로 지정해준다. view는 tensor를 바꾸는 것

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid.to(x.device)) * stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            y = y.view(bs, -1, self.no)

        return x if self.training else y

    @staticmethod
    def _make_grid(nx=20, ny=20): # grid 생성 후 bounding box를 표기하기위함.
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module): # darknet 클래스 정의
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path) # parse_config를 module_defs로 정의
        self.hyperparams, self.module_list = create_modules(self.module_defs) # module_defs안의 hyperparameters에서 가져온다.
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)] # 
        self.img_size = img_size # img size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32) # config 파일에서 가져온 parameters를 생성자에 대입시켜준다.

    def forward(self, x): # 이전에 정의했던 층들을 가져와 yolo의 모듈을 만들어준다.
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x) # convolutional, upsample, maxpool 층이 나오면 forward propogation 진행
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1) # route가 나오면 module_def 안의 layers들을 torch.cat함수로 연결한다.
            elif module_def["type"] == "shortcut": # shortcut이 나오면 residual층 진행
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo": # yolo알고리즘 진행
                x = module[0](x)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path): # pretrained weight를 가져와서 학습전에 넣어준다. 
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional": # convoultional층일 때 
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1] # batch normalization 층의 layer를 가져옴
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b) # bn_layer의 bias가져와서 copy.
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w) # bn_layer의 weight가져와서 copy.
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm) # bn_layer의 mean가져와서 copy.
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv) # bn_layer의 variance가져와서 copy.
                    ptr += num_b # batch normalization의 각 평균과 분산, bias, weight를 가지고 ptr에 추가 시켜줌
                else:
                    # conv와 bias를 가져옴
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # conv와 weight를 가져옴
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1): # darknet으로 학습중에 weight들을 저장하기 위한 함수
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp) # 파일에 parameter들을 저장하기 위한 단계

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0] # module의 첫번째
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1] 
                    bn_layer.bias.data.cpu().numpy().tofile(fp) # bn_layer의 bias를 file저장
                    bn_layer.weight.data.cpu().numpy().tofile(fp) # bn_layer의 weight를 file저장
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp) # bn_layer의 평균를 file저장
                    bn_layer.running_var.data.cpu().numpy().tofile(fp) # bn_layer의 분산를 file저장
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close() # 파일 수정 및 저장 완료
