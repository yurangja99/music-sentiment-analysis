import torch
import librosa
import numpy as np
import time
import argparse
from sklearn import preprocessing
from playsound import playsound

from user_values import uv
from model import ResidualBlock, BottleneckResidualBlock, Network

parser = argparse.ArgumentParser()
parser.add_argument("model_version", type=str, help="name of the model (except .pt)")
parser.add_argument("audio_path", type=str, help="path to the audio file")
parser.add_argument("--chunk_size", type=int, default=128, help="size of each chunk. default to 128 (3 secs)")
parser.add_argument("--print_realtime", action="store_true", help="if specified, print the result realtime")
parser.add_argument("--max_tags", type=int, default=3, help="max number of tags")
parser.add_argument("--v_mean", type=float, default=5.4823, help="mean of V")
parser.add_argument("--a_mean", type=float, default=4.3053, help="mean of A")
parser.add_argument("--d_mean", type=float, default=5.2389, help="mean of D")
parser.add_argument("--v_std", type=float, default=1.5542, help="std of V")
parser.add_argument("--a_std", type=float, default=1.1655, help="std of A")
parser.add_argument("--d_std", type=float, default=1.1763, help="std of D")
args = parser.parse_args()

# list of emotion tags
emotion_tag_list = uv.EMOTION_TAG_LIST

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
ckpt = torch.load(f"{args.model_version}.pt")
model = Network(
  num_tags=len(emotion_tag_list),
  use_mel=ckpt["use_mel"],
  use_mfcc=ckpt["use_mfcc"],
  use_chroma=ckpt["use_chroma"],
  block=ResidualBlock if ckpt["block_name"] == "basic" else BottleneckResidualBlock,
  num_blocks=ckpt["num_blocks"]
).to(device=device)
model.load_state_dict(ckpt["model"])

# print model info
print(f"Model info:")
print(f"use mel: {model.use_mel}")
print(f"use mfcc: {model.use_mfcc}")
print(f"use chroma: {model.use_chroma}")
print(f"block: {model.block}")
print(f"number of blocks: {model.num_blocks}")
print(f"for {model.num_tags} tags")
print()

# load audio file
y, sr = librosa.load(args.audio_path)

# mel-spectrogram
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.amplitude_to_db(mel, ref=np.max)
mel = preprocessing.minmax_scale(mel)

# MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr)
mfcc = preprocessing.minmax_scale(mfcc)

# chroma frequencies
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
chroma = preprocessing.minmax_scale(chroma)

# pad to make multiple of chunk_size
if mel.shape[1] % args.chunk_size != 0:
  pad = args.chunk_size - (mel.shape[1] % args.chunk_size)
  mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
  mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
  chroma = np.pad(chroma, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

# chunk input data
chunk_num = mel.shape[1] // args.chunk_size
mel = np.array(np.split(mel, chunk_num, axis=1))
mfcc = np.array(np.split(mfcc, chunk_num, axis=1))
chroma = np.array(np.split(chroma, chunk_num, axis=1))

# print audio info
print(f"Audio info:")
print(f"path: {args.audio_path}")
print(f"length: {y.shape[0] / sr} seconds")
print(f"number of chunks: {chunk_num}")
print(f"mel: {mel.shape}")
print(f"mfcc: {mfcc.shape}")
print(f"chroma: {chroma.shape}")
print()

# inference
model.eval()
mel = torch.tensor(mel, dtype=torch.float32, device=device).unsqueeze(1)
mfcc = torch.tensor(mfcc, dtype=torch.float32, device=device).unsqueeze(1)
chroma = torch.tensor(chroma, dtype=torch.float32, device=device).unsqueeze(1)
with torch.no_grad():
  vad_pred, tag_pred = model(mel, mfcc, chroma)
tag_pred = torch.nn.Sigmoid()(tag_pred)

# get VAD vector
mean = torch.tensor([args.v_mean, args.a_mean, args.d_mean], dtype=torch.float32, device=device)
std = torch.tensor([args.v_std, args.a_std, args.d_std], dtype=torch.float32, device=device)
vad = vad_pred * std + mean
vectors = []
for cidx in range(chunk_num):
  v, a, d = vad[cidx].tolist()
  vectors.append((round(v, 2), round(a, 2), round(d, 2)))

# get tags
tags = []
for cidx in range(chunk_num):
  temp = [(round(tag_pred[cidx][idx].item(), 2), emotion_tag_list[idx]) for idx in range(len(emotion_tag_list)) if tag_pred[cidx][idx] >= 0.5]
  temp = sorted(temp, reverse=True)
  temp = temp[:args.max_tags]
  tags.append(temp)

# print inference result
if args.print_realtime:
  print(f"Inference Result (enter to start):")
  input()
  playsound(args.audio_path, block=False)
  for cidx in range(chunk_num):
    print(f"VAD: {vectors[cidx]}\t\tTags: {tags[cidx]}")
    time.sleep(args.chunk_size * 512 / 22050)
else:
  print(f"Inference Result:")
  for cidx in range(chunk_num):
    print(f"VAD: {vectors[cidx]}\t\tTags: {tags[cidx]}")
