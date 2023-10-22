import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ResidualBlock(nn.Module):
  """
  Basic Residual Block
  """
  def __init__(self, in_channels: int, out_channels: int, stride: int=1, groups: int=1):
    """
    Initialize basic residual block.

    Args:
        - in_channels (int): number of channels of input tensor
        - out_channels (int): number of channels of output tensor
        - stride (int, optional): CNN stride. Defaults to 1.
        - groups (int, optional): groups for CNN. Defaults to 1.
    """
    super(ResidualBlock, self).__init__()
    
    assert in_channels % groups == 0
    assert out_channels % groups == 0
    
    # convolution block
    self.layer = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False),
      nn.BatchNorm2d(out_channels),
    )
    
    # shortcut
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.shortcut = nn.Identity()
  
  def forward(self, x: torch.Tensor):
    """
    Calculate H(x) = F(x) + x. 
    - H(x): output of this function. 
    - F(x): output of convolution block.
    - x: input tensor itself, or shortcut if shape differs to F(x). 

    Args:
        - x (torch.Tensor): input tensor

    Returns:
        - torch.Tensor: output of the residual block
    """
    return F.relu(self.layer(x) + self.shortcut(x))

class BottleneckResidualBlock(nn.Module):
  """
  Bottleneck Residual Block
  """
  def __init__(self, in_channels: int, out_channels: int, stride: int=1, groups: int=1):
    """
    Initialize bottleneck residual block.

    Args:
        - in_channels (int): number of channels of input tensor
        - out_channels (int): number of channels of output tensor. should be multiple of 4. 
        - stride (int, optional): CNN stride. Defaults to 1.
        - groups (int, optional): groups for CNN. Defaults to 1.
    """
    super(BottleneckResidualBlock, self).__init__()
    
    assert in_channels % groups == 0
    assert out_channels % (4 * groups) == 0
    
    # convolution block
    self.layer = nn.Sequential(
      nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0, groups=groups, bias=False),
      nn.BatchNorm2d(out_channels // 4),
      nn.ReLU(),
      nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
      nn.BatchNorm2d(out_channels // 4),
      nn.ReLU(),
      nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False),
      nn.BatchNorm2d(out_channels)
    )
    
    # shortcut
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.shortcut = nn.Identity()
        
  def forward(self, x):
    """
    Calculate H(x) = F(x) + x. 
    - H(x): output of this function. 
    - F(x): output of convolution block.
    - x: input tensor itself, or shortcut if shape differs to F(x). 

    Args:
        - x (torch.Tensor): input tensor

    Returns:
        - torch.Tensor: output of the residual block
    """
    return F.relu(self.layer(x) + self.shortcut(x))

class Network(nn.Module):
  """
  Music-Sentiment-Analysis model
  
  It uses features of given audio, which can be considered as image - visualized audio. 
  It predicts sentiment of it as VAD vector, and emotion tags that well explains the given audio. 
  
  The structure of this network is based on Network by Google. 
  
  One of the input options must be available. 
  
  Input:
      - Mel-Spectrogram (torch.Tensor, optional)
      - MFCCs (torch.Tensor, optional)
      - Chroma Frequencies (torch.Tensor, optional)
  
  Output:
      - VAD vector (torch.Tensor): (B, 3) vector represents sentiment of given audios
      - Emotion tags (torch.Tensor): (B, num_tags) vector classifies emotion tags whether they well explain the given audios or not
  """
  def __init__(
    self,
    num_tags: int,
    use_mel: bool=True,
    use_mfcc: bool=True,
    use_chroma: bool=True,
    block: type=ResidualBlock,
    num_blocks: list=[1, 1, 1, 1, 1, 1]
  ):
    """
    Initialize a model predicts sentiment and emotion tags of given audio.

    Args:
        - num_tags (int): number of emotional tags
        - use_mel (bool, optional): use Mel-Spectrogram as input. Defaults to True.
        - use_mfcc (bool, optional): use MFCCs as input. Defaults to True.
        - use_chroma (bool, optional): use Chroma Frequencies as input. Defaults to True.
        - block (type, optional): ResidualBlock or BottleneckResidualBlock. Defaults to ResidualBlock.
        - num_blocks (list, optional): length of the list must be 6. Some options - refered by ResNet structure - are recommended. 
            - [2, 2, 2, 2, 2, 2] with ResidualBlock
            - [3, 4, 6, 6, 4, 3] with ResidualBlock
            - [3, 4, 6, 6, 4, 3] with BottleneckResidualBlock
            - [3, 4, 23, 23, 4, 3] with BottleneckResidualBlock
            - [3, 8, 36, 36, 8, 3] with BottleneckResidualBlock
    """
    super(Network, self).__init__()
    
    # assert input parameters
    assert use_mel or use_mfcc or use_chroma
    assert len(num_blocks) == 6
    
    self.num_tags = num_tags
    self.use_mel = use_mel
    self.use_mfcc = use_mfcc
    self.use_chroma = use_chroma
    self.block = block
    self.num_blocks = num_blocks
    
    self.groups = use_mel + use_mfcc + use_chroma
    
    # feature extraction model
    self.feat_model = nn.Sequential(
      nn.Conv2d(3, 64 * self.groups, kernel_size=3, stride=1, padding=1, groups=self.groups, bias=False),
      nn.BatchNorm2d(64 * self.groups),
      self.__make_block(block, 64 * self.groups, 64 * self.groups, num_blocks[0], stride=1),
      self.__make_block(block, 64 * self.groups, 128 * self.groups, num_blocks[1], stride=2),
      self.__make_block(block, 128 * self.groups, 128 * self.groups, num_blocks[2], stride=2),
      self.__make_block(block, 128 * self.groups, 256 * self.groups, num_blocks[3], stride=2),
      self.__make_block(block, 256 * self.groups, 256 * self.groups, num_blocks[4], stride=2),
      self.__make_block(block, 256 * self.groups, 512 * self.groups, num_blocks[5], stride=2),
      nn.AvgPool2d(kernel_size=4),
      nn.Flatten()
    )
    
    # vad regression model
    self.sentiment_model = nn.Sequential(
      nn.Linear(512 * self.groups, 256),
      nn.ReLU(),
      nn.Linear(256, 3)
    )
    
    # emotion tagging model
    self.emotion_tagging_model = nn.Sequential(
      nn.Linear(512 * self.groups, 256),
      nn.ReLU(),
      nn.Linear(256, num_tags),
      nn.Sigmoid() # sigmoid to classify
    )
  
  def __make_block(
    self, 
    block: type, 
    in_channels: int,
    out_channels: int,
    num_blocks: int=1, 
    stride: int=1
  ):
    """
    Make a block consists of several residual blocks. 

    Args:
        - block (type): ResidualBlock or BottleneckResidualBlock
        - in_channels (int): number of channels of input tensor
        - out_channels (int): number of channels of output tensor
        - num_blocks (int): number of blocks. Defaults to 1.
        - stride (int): CNN stride. Defaults to 1.
    
    Return:
        - nn.Module: block including several residual blocks
    """
    layers = [block(in_channels, out_channels, stride=stride, groups=self.groups)]
    for _ in range(num_blocks - 1):
      layers.append(block(out_channels, out_channels, stride=1, groups=self.groups))
    return nn.Sequential(*layers)
  
  def forward(
    self, 
    mel: torch.Tensor | None,
    mfcc: torch.Tensor | None, 
    chroma: torch.Tensor | None
  ):
    # cat (B, 1, 128, 128) tensors to (B, 3, 128, 128)
    x = []
    if self.use_mel:
      x.append(mel)
    if self.use_mfcc:
      x.append(F.interpolate(mfcc, (128, 128), mode="bilinear"))
    if self.use_chroma:
      x.append(F.interpolate(chroma, (128, 128), mode="bilinear"))
    x = torch.cat(x, dim=1)
    feat = self.feat_model(x)
    vad = self.sentiment_model(feat)
    tags = self.emotion_tagging_model(feat)
    return vad, tags

  def save(self, path: str):
    """
    Save checkpoint for this model. 

    Args:
        - path (str): path to the checkpoint
    """
    torch.save({
      "model": self.state_dict(),
      "num_tags": self.num_tags,
      "use_mel": self.use_mel,
      "use_mfcc": self.use_mfcc,
      "use_chroma": self.use_chroma,
      "block_name": "basic" if self.block == ResidualBlock else "bottleneck",
      "num_blocks": self.num_blocks
    }, path)
  
  @staticmethod
  def load(path: str):
    """
    Returns a new model from the given path. 
    
    Args:
        - path (str): path to the checkpoint
    
    Return:
        - Network: loaded network.
    """
    ckpt = torch.load(path)
    network = Network(
      num_tags=ckpt["num_tags"],
      use_mel=ckpt["use_mel"],
      use_mfcc=ckpt["use_mfcc"],
      use_chroma=ckpt["use_chroma"],
      block=ResidualBlock if ckpt["block_name"] == "basic" else BottleneckResidualBlock,
      num_blocks=ckpt["num_blocks"]
    )
    network.load_state_dict(ckpt["model"])
    return network

  def print_summary(self):
    """
    Print summary of this model.
    """
    x = []
    if self.use_mel:
      x.append((32, 1, 128, 128))
    if self.use_mfcc:
      x.append((32, 1, 128, 20))
    if self.use_chroma:
      x.append((32, 1, 128, 12))
    summary(self, x, depth=10)
