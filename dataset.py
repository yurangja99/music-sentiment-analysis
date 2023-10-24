import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  """
  Dataset consists of meta-data of audio and numpy tensors. 
  """
  def __init__(
    self, 
    npz_path_list: list, 
    emotion_tag_list: list, 
    normalize_vad: bool=True, 
    device: str="cuda"
  ):
    """
    Initialize data from number of npz files. 
    
    Audio features such as Mel-Spectrogram, MFCCs, and Chroma will be cropped to have a length of 128 which means 3-second in audio. 

    Args:
        - npz_path_list (list): list of paths to npz files
        - emotion_tag_list (list): list of interested emotion tags
        - normalize_vad (bool): whether normalize vad label, or not. Defaults to True.
        - device (str): "cpu" or "cuda". Defaults to "cuda".
    """
    # emotion tag index
    emotion_tag_cnt = len(emotion_tag_list)
    emotion_tag_idx = dict((t, i) for (i, t) in enumerate(emotion_tag_list))
    
    self.device = device
    
    # dataset in numpy array form
    self.name_array = []
    self.sid_array = []
    self.mel_array = []
    self.mfcc_array = []
    self.chroma_array = []
    self.vad_array = []
    self.tag_array = []
    
    print(f"Start loading dataset from {len(npz_path_list)} files...")
    
    # append each file's data to full dataset
    for npz_path in tqdm(npz_path_list):
      data = np.load(npz_path, allow_pickle=True)
      
      # keys: name, sid, mel, mfcc, chroma, vad, tag
      self.name_array.append(data["name"])
      self.sid_array.append(data["sid"])
      self.mel_array.append(torch.tensor(data["mel"], dtype=torch.float32, device=self.device))
      self.mfcc_array.append(torch.tensor(data["mfcc"], dtype=torch.float32, device=self.device))
      self.chroma_array.append(torch.tensor(data["chroma"], dtype=torch.float32, device=self.device))
      self.vad_array.append(torch.tensor(data["vad"], dtype=torch.float32, device=self.device))
      
      # generate tag vector
      is_interesting_tag = np.vectorize(lambda t: t in emotion_tag_list, otypes=[bool])
      find_tag_idx = np.vectorize(lambda t: emotion_tag_idx[t], otypes=[int])
      idxs = [find_tag_idx(tags[is_interesting_tag(tags)]) for tags in data["tag"]]
      tags = [np.sum(np.eye(emotion_tag_cnt)[idx], axis=0) for idx in idxs]
      tags = torch.tensor(np.array(tags, dtype=np.float32), dtype=torch.float32, device=self.device)
      self.tag_array.append(tags)
    
    self.name_array = np.concatenate(self.name_array, axis=0)
    self.sid_array = np.concatenate(self.sid_array, axis=0)
    self.mel_array = torch.cat(self.mel_array, dim=0)
    self.mfcc_array = torch.cat(self.mfcc_array, dim=0)
    self.chroma_array = torch.cat(self.chroma_array, dim=0)
    self.vad_array = torch.cat(self.vad_array, dim=0)
    self.tag_array = torch.cat(self.tag_array, dim=0)
    
    # validation
    assert self.name_array.shape[0] == self.sid_array.shape[0]
    assert self.name_array.shape[0] == self.mel_array.shape[0]
    assert self.name_array.shape[0] == self.mfcc_array.shape[0]
    assert self.name_array.shape[0] == self.chroma_array.shape[0]
    assert self.name_array.shape[0] == self.vad_array.shape[0]
    assert self.name_array.shape[0] == self.tag_array.shape[0]
    
    # if normalize mode, get mean and std of vad. 
    self.normalize_vad = normalize_vad
    self.vad_mean = torch.mean(self.vad_array, dim=0)
    self.vad_std = torch.std(self.vad_array, dim=0)
    
    print(f"Loading dataset done!: {len(self)} items...")
  
  def __len__(self):
    """
    Return size of the dataset. 
    
    Return:
        - int: length of the dataset
    """
    return self.name_array.shape[0]
  
  def __getitem__(self, idx: int):
    """
    Return data at index idx. 

    Args:
        - idx (int): index.
    
    Return:
        - tuple: list of torch tensors
    """
    # validation
    assert self.mel_array[idx].shape[1] == self.mfcc_array[idx].shape[1]
    assert self.mel_array[idx].shape[1] == self.chroma_array[idx].shape[1]
    
    # randomly crop 128 in input data
    start_idx = random.randint(0, self.mel_array[idx].shape[1] - 128)
    
    return \
      self.name_array[idx], \
      self.sid_array[idx], \
      self.mel_array[idx][:, start_idx:start_idx + 128].unsqueeze(0), \
      self.mfcc_array[idx][:, start_idx:start_idx + 128].unsqueeze(0), \
      self.chroma_array[idx][:, start_idx:start_idx + 128].unsqueeze(0), \
      (self.vad_array[idx] - self.vad_mean) / self.vad_std if self.normalize_vad else self.vad_array[idx], \
      self.tag_array[idx]
