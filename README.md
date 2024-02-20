# music-sentiment-analysis

Music Sentiment Analyzer (MSA) predicts real-time sentiment of given music as VAD:
- **Valence**: the pleasantness of a stimulus
- **Arousal**: the intensity of emotion provoked by a stimulus
- **Dominance**: the degree of control exerted by a stimulus

It also provides appropriate tags for the song, according to its mood. 
There are 276 tags those represents the song, and you can check full list of it in [`train.py`](./train.py).
```
emotion_tag_list = [
  'acerbic', 'aggressive', 'agreeable', 'airy', 'ambitious', 'amiable', 'angry', 'angst-ridden', 
  'animated', 'anxious', 'apocalyptic', 'athletic', 'atmospheric', 'austere', 'autumnal', 
  ...
  'unsettling', 'uplifting', 'urgent', 'virile', 'visceral', 'volatile', 'warm', 'weary', 
  'whimsical', 'wintry', 'wistful', 'witty', 'wry', 'yearning'
]
```


## How to run

Codes of this repository were implemented on:
- windows 10
- python 3.11.5
- torch 2.1.0+cu118

First, install required packages with conda. 
If error about torch occurred, delete torch, torchvision, and torchaudio in [`environment.yaml`](./environment.yaml) and retry. 
```
conda env create -f environment.yaml
conda activate msa
```

You have to get [MuSe Dataset](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset) to generate your own dataset. 
You also have to log in Spotify API id and get CID and SECRET [here](https://developer.spotify.com).

Next, open [`user_values.py`](./user_values.py) and type path to your csv file, and Spotify API information. 
- `MUSE_CSV_PATH`: path to your MuSe csv file
- `SPOTIFY_CID`: Spotify API CID
- `SPOTIFY_SECRET`: Spotify API SECRET

Run [`generate_dataset.py`](./generate_dataset.py) to preprocess your own dataset. 
After running the command, you can see some .npy and .npz files in `data` directory. 
You can edit some parameters in editable zone if you can. 
```
python generate_dataset.py
```

To train the model, run [`train.py`](./train.py). 
Hyper-parameters such as number of epochs, learning rate, model structure, and input data can be editted in the editable zone. 
```
python train.py
```

There are some example training results in [Appendix C](#appendix-c-trianing-examples). 


## Dataset

The example data is shown in [Appendix A](#appendix-a-example-of-dataset). 

I used [MuSe Dataset](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset) that contains sentiment information for 90,001 songs. 

It contains Spotify ID of songs, whose audios are available by using [Spotipy API](https://spotipy.readthedocs.io/en/2.22.1/?highlight=analysis#). However, there are some rows with no spotify id or no preview mp3, 34,921 songs are actually used for this project. 

After pre-processing the dataset, input and output of my model are:
- **input**
    - Mel-spectrogram
        ![](./assets/example_mel.png)
    - Mel-Frequency Cepstral Coefficients (MFCCs)
        ![](./assets/example_mfcc.png)
    - Chroma Frequencies
        ![](./assets/example_chroma.png)
- **output**
    - VAD vector as (B, 3) `torch.Tensor`
        - `[[2.09, 6.18, 1.4]]` means the audio has high **Arousal**, while it has low **Valence** and **Dominance**. 
    - list of emotion tags as (B, num_tags) `torch.Tensor`. Each tag output is activated with `Sigmoid()`, so we can decide whether it matches the given audio or not. 
        - `[[1., 0., 1., 0.]]` for emotion tag dictionary `["reckless", "innocent", "confident", "serious]` means the audio tends to be reckless and confident, and not innocent and serious. 

Because dataset preprocessing task requires a lot of time, I implemented [`generate_dataset.py`](./generate_dataset.py) to save chunk of the full dataset. 
After running the code, there would be some .npy and .npz files instead of one huge file. 
The files are combined to a full dataset during training. 


## Model

The precise structure of my network is shown in [Appendix B](#appendix-b-structure-of-network). 

I refered to the structure of [ResNet](https://arxiv.org/abs/1512.03385), which is specialized to image classification task. 

I used residual blocks and bottleneck residual blocks. (refer to [`model.py`](./model.py))
CNN layers in the blocks are grouped so that features from Mel-Spectrogram don't interact with features from MFCCs during feature extraction. ([Concepts of group in CNN](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html))
- If we use all of Mel-Spectrogram, MFCCs, and Chroma Frequencies, CNN layers are grouped by 3. 
- If we use two of them, CNN layers are grouped by 2. 
- If we use only one of them, CNN layers act like usual. 

Whole structure of the network is shown below (we use all inputs):

![](./assets/model.png)

The residual block is one of basic residual block or bottleneck residual block. 

Basic residual block and bottleneck residual block are implemented as shown below. (`k`: kernel size, `s`: stride, `p`: padding, `g`: groups)
Bottleneck residual block reduces number of channels during 3x3 convolution, so it can reduce parameters of the block.

![](./assets/residual_block_1.png)

To reduce width and height of the feature, the first residual blocks of block-group (dashed-line border in the model figure) have stride = 2. 
Then, residual block should use short-cut convolution layers instead of identity function, because the output width and height reduces. 

![](./assets/residual_block_2.png)


## Appendix

### Appendix A: example of dataset

Example of dataset created by [`generate_dataset.py`](./generate_dataset.py). 
```
from dataset import CustomDataset
dataset = CustomDataset(
  path_list=[
    "data/generated_dataset_from_0_to_100",
  ],
  normalize_vad=True,
  device="cuda"
)

print("Len:", len(dataset))

elem = dataset[0]

print("Example:")
print(elem[0])
print(elem[1])
print(elem[2].shape)
print(elem[3].shape)
print(elem[4].shape)
print(elem[5], elem[5].shape)
print(elem[6], elem[6].shape)


Output:
Len: 3050
Example:
Die MF Die
5bU4KX47KqtDKKaLM4QCzh
torch.Size([1, 128, 128])
torch.Size([1, 20, 128])
torch.Size([1, 12, 128])
tensor([-0.4578,  0.1240,  0.6577], device='cuda:0') torch.Size([3])
tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.], device='cuda:0') torch.Size([276])
```

### Appendix B: structure of network

The structure of network can be printed using the script:
```
from model import ResidualBlock, BottleneckResidualBlock, Network
net = Network(
  num_tags=50,
  use_mel=True,
  use_mfcc=True,
  use_chroma=True,
  block=ResidualBlock,
  num_blocks=[2, 2, 2, 2, 2, 2]
)
net.print_summary()

Output:
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
Network                                       [32, 3]                   --
├─Sequential: 1-1                             [32, 1536]                --
│    └─Conv2d: 2-1                            [32, 192, 128, 128]       1,728
│    └─BatchNorm2d: 2-2                       [32, 192, 128, 128]       384
│    └─Sequential: 2-3                        [32, 192, 128, 128]       --
│    │    └─ResidualBlock: 3-1                [32, 192, 128, 128]       --
│    │    │    └─Sequential: 4-1              [32, 192, 128, 128]       --
│    │    │    │    └─Conv2d: 5-1             [32, 192, 128, 128]       110,592
│    │    │    │    └─BatchNorm2d: 5-2        [32, 192, 128, 128]       384
│    │    │    │    └─ReLU: 5-3               [32, 192, 128, 128]       --
│    │    │    │    └─Conv2d: 5-4             [32, 192, 128, 128]       110,592
│    │    │    │    └─BatchNorm2d: 5-5        [32, 192, 128, 128]       384
│    │    │    └─Identity: 4-2                [32, 192, 128, 128]       --
│    │    └─ResidualBlock: 3-2                [32, 192, 128, 128]       --
│    │    │    └─Sequential: 4-3              [32, 192, 128, 128]       --
│    │    │    │    └─Conv2d: 5-6             [32, 192, 128, 128]       110,592
│    │    │    │    └─BatchNorm2d: 5-7        [32, 192, 128, 128]       384
│    │    │    │    └─ReLU: 5-8               [32, 192, 128, 128]       --
│    │    │    │    └─Conv2d: 5-9             [32, 192, 128, 128]       110,592
│    │    │    │    └─BatchNorm2d: 5-10       [32, 192, 128, 128]       384
│    │    │    └─Identity: 4-4                [32, 192, 128, 128]       --
│    └─Sequential: 2-4                        [32, 384, 64, 64]         --
│    │    └─ResidualBlock: 3-3                [32, 384, 64, 64]         --
│    │    │    └─Sequential: 4-5              [32, 384, 64, 64]         --
│    │    │    │    └─Conv2d: 5-11            [32, 384, 64, 64]         221,184
│    │    │    │    └─BatchNorm2d: 5-12       [32, 384, 64, 64]         768
│    │    │    │    └─ReLU: 5-13              [32, 384, 64, 64]         --
│    │    │    │    └─Conv2d: 5-14            [32, 384, 64, 64]         442,368
│    │    │    │    └─BatchNorm2d: 5-15       [32, 384, 64, 64]         768
│    │    │    └─Sequential: 4-6              [32, 384, 64, 64]         --
│    │    │    │    └─Conv2d: 5-16            [32, 384, 64, 64]         24,576
│    │    │    │    └─BatchNorm2d: 5-17       [32, 384, 64, 64]         768
│    │    └─ResidualBlock: 3-4                [32, 384, 64, 64]         --
│    │    │    └─Sequential: 4-7              [32, 384, 64, 64]         --
│    │    │    │    └─Conv2d: 5-18            [32, 384, 64, 64]         442,368
│    │    │    │    └─BatchNorm2d: 5-19       [32, 384, 64, 64]         768
│    │    │    │    └─ReLU: 5-20              [32, 384, 64, 64]         --
│    │    │    │    └─Conv2d: 5-21            [32, 384, 64, 64]         442,368
│    │    │    │    └─BatchNorm2d: 5-22       [32, 384, 64, 64]         768
│    │    │    └─Identity: 4-8                [32, 384, 64, 64]         --
│    └─Sequential: 2-5                        [32, 384, 32, 32]         --
│    │    └─ResidualBlock: 3-5                [32, 384, 32, 32]         --
│    │    │    └─Sequential: 4-9              [32, 384, 32, 32]         --
│    │    │    │    └─Conv2d: 5-23            [32, 384, 32, 32]         442,368
│    │    │    │    └─BatchNorm2d: 5-24       [32, 384, 32, 32]         768
│    │    │    │    └─ReLU: 5-25              [32, 384, 32, 32]         --
│    │    │    │    └─Conv2d: 5-26            [32, 384, 32, 32]         442,368
│    │    │    │    └─BatchNorm2d: 5-27       [32, 384, 32, 32]         768
│    │    │    └─Sequential: 4-10             [32, 384, 32, 32]         --
│    │    │    │    └─Conv2d: 5-28            [32, 384, 32, 32]         49,152
│    │    │    │    └─BatchNorm2d: 5-29       [32, 384, 32, 32]         768
│    │    └─ResidualBlock: 3-6                [32, 384, 32, 32]         --
│    │    │    └─Sequential: 4-11             [32, 384, 32, 32]         --
│    │    │    │    └─Conv2d: 5-30            [32, 384, 32, 32]         442,368
│    │    │    │    └─BatchNorm2d: 5-31       [32, 384, 32, 32]         768
│    │    │    │    └─ReLU: 5-32              [32, 384, 32, 32]         --
│    │    │    │    └─Conv2d: 5-33            [32, 384, 32, 32]         442,368
│    │    │    │    └─BatchNorm2d: 5-34       [32, 384, 32, 32]         768
│    │    │    └─Identity: 4-12               [32, 384, 32, 32]         --
│    └─Sequential: 2-6                        [32, 768, 16, 16]         --
│    │    └─ResidualBlock: 3-7                [32, 768, 16, 16]         --
│    │    │    └─Sequential: 4-13             [32, 768, 16, 16]         --
│    │    │    │    └─Conv2d: 5-35            [32, 768, 16, 16]         884,736
│    │    │    │    └─BatchNorm2d: 5-36       [32, 768, 16, 16]         1,536
│    │    │    │    └─ReLU: 5-37              [32, 768, 16, 16]         --
│    │    │    │    └─Conv2d: 5-38            [32, 768, 16, 16]         1,769,472
│    │    │    │    └─BatchNorm2d: 5-39       [32, 768, 16, 16]         1,536
│    │    │    └─Sequential: 4-14             [32, 768, 16, 16]         --
│    │    │    │    └─Conv2d: 5-40            [32, 768, 16, 16]         98,304
│    │    │    │    └─BatchNorm2d: 5-41       [32, 768, 16, 16]         1,536
│    │    └─ResidualBlock: 3-8                [32, 768, 16, 16]         --
│    │    │    └─Sequential: 4-15             [32, 768, 16, 16]         --
│    │    │    │    └─Conv2d: 5-42            [32, 768, 16, 16]         1,769,472
│    │    │    │    └─BatchNorm2d: 5-43       [32, 768, 16, 16]         1,536
│    │    │    │    └─ReLU: 5-44              [32, 768, 16, 16]         --
│    │    │    │    └─Conv2d: 5-45            [32, 768, 16, 16]         1,769,472
│    │    │    │    └─BatchNorm2d: 5-46       [32, 768, 16, 16]         1,536
│    │    │    └─Identity: 4-16               [32, 768, 16, 16]         --
│    └─Sequential: 2-7                        [32, 768, 8, 8]           --
│    │    └─ResidualBlock: 3-9                [32, 768, 8, 8]           --
│    │    │    └─Sequential: 4-17             [32, 768, 8, 8]           --
│    │    │    │    └─Conv2d: 5-47            [32, 768, 8, 8]           1,769,472
│    │    │    │    └─BatchNorm2d: 5-48       [32, 768, 8, 8]           1,536
│    │    │    │    └─ReLU: 5-49              [32, 768, 8, 8]           --
│    │    │    │    └─Conv2d: 5-50            [32, 768, 8, 8]           1,769,472
│    │    │    │    └─BatchNorm2d: 5-51       [32, 768, 8, 8]           1,536
│    │    │    └─Sequential: 4-18             [32, 768, 8, 8]           --
│    │    │    │    └─Conv2d: 5-52            [32, 768, 8, 8]           196,608
│    │    │    │    └─BatchNorm2d: 5-53       [32, 768, 8, 8]           1,536
│    │    └─ResidualBlock: 3-10               [32, 768, 8, 8]           --
│    │    │    └─Sequential: 4-19             [32, 768, 8, 8]           --
│    │    │    │    └─Conv2d: 5-54            [32, 768, 8, 8]           1,769,472
│    │    │    │    └─BatchNorm2d: 5-55       [32, 768, 8, 8]           1,536
│    │    │    │    └─ReLU: 5-56              [32, 768, 8, 8]           --
│    │    │    │    └─Conv2d: 5-57            [32, 768, 8, 8]           1,769,472
│    │    │    │    └─BatchNorm2d: 5-58       [32, 768, 8, 8]           1,536
│    │    │    └─Identity: 4-20               [32, 768, 8, 8]           --
│    └─Sequential: 2-8                        [32, 1536, 4, 4]          --
│    │    └─ResidualBlock: 3-11               [32, 1536, 4, 4]          --
│    │    │    └─Sequential: 4-21             [32, 1536, 4, 4]          --
│    │    │    │    └─Conv2d: 5-59            [32, 1536, 4, 4]          3,538,944
│    │    │    │    └─BatchNorm2d: 5-60       [32, 1536, 4, 4]          3,072
│    │    │    │    └─ReLU: 5-61              [32, 1536, 4, 4]          --
│    │    │    │    └─Conv2d: 5-62            [32, 1536, 4, 4]          7,077,888
│    │    │    │    └─BatchNorm2d: 5-63       [32, 1536, 4, 4]          3,072
│    │    │    └─Sequential: 4-22             [32, 1536, 4, 4]          --
│    │    │    │    └─Conv2d: 5-64            [32, 1536, 4, 4]          393,216
│    │    │    │    └─BatchNorm2d: 5-65       [32, 1536, 4, 4]          3,072
│    │    └─ResidualBlock: 3-12               [32, 1536, 4, 4]          --
│    │    │    └─Sequential: 4-23             [32, 1536, 4, 4]          --
│    │    │    │    └─Conv2d: 5-66            [32, 1536, 4, 4]          7,077,888
│    │    │    │    └─BatchNorm2d: 5-67       [32, 1536, 4, 4]          3,072
│    │    │    │    └─ReLU: 5-68              [32, 1536, 4, 4]          --
│    │    │    │    └─Conv2d: 5-69            [32, 1536, 4, 4]          7,077,888
│    │    │    │    └─BatchNorm2d: 5-70       [32, 1536, 4, 4]          3,072
│    │    │    └─Identity: 4-24               [32, 1536, 4, 4]          --
│    └─AvgPool2d: 2-9                         [32, 1536, 1, 1]          --
│    └─Flatten: 2-10                          [32, 1536]                --
├─Sequential: 1-2                             [32, 3]                   --
│    └─Linear: 2-11                           [32, 256]                 393,472
│    └─ReLU: 2-12                             [32, 256]                 --
│    └─Linear: 2-13                           [32, 3]                   771
├─Sequential: 1-3                             [32, 50]                  --
│    └─Linear: 2-14                           [32, 256]                 393,472
│    └─ReLU: 2-15                             [32, 256]                 --
│    └─Linear: 2-16                           [32, 50]                  12,850
│    └─Sigmoid: 2-17                          [32, 50]                  --
===============================================================================================
Total params: 43,408,245
Trainable params: 43,408,245
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 577.93
===============================================================================================
Input size (MB): 2.62
Forward/backward pass size (MB): 13778.43
Params size (MB): 173.63
Estimated Total Size (MB): 13954.69
===============================================================================================
```

### Appendix C: trianing examples
full dataset으로 학습 후 작성
