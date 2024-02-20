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
    path_list: list, 
    normalize_vad: bool=True, 
    device: str="cuda"
  ):
    """
    Initialize data from number of npz files. 
    
    Large data such as Mel-Spectrogram, MFCCs, and Chroma frequencies are stored in disk and read by np.memmap(). 

    Args:
        - path_list (list): list of paths to .dat files
        - normalize_vad (bool): whether normalize vad label, or not. Defaults to True.
        - device (str): "cpu" or "cuda". Defaults to "cuda".
    """
    # device
    self.device = device
    
    # npz index and data index
    self.dat_idx = []
    self.data_idx = []
    
    # data
    self.name_list = []
    self.sid_list = []
    self.mel_list = []
    self.mfcc_list = []
    self.chroma_list = []
    self.vad_list = []
    self.tag_list = []
    
    print(f"Start loading dataset from {len(path_list)} files...")
    
    # append each file's data to full dataset
    for didx, dat_path in enumerate(tqdm(path_list)):
      # append memmapped np array
      name_and_sid = np.load(dat_path + "_name_sid.npz", allow_pickle=True)
      self.name_list.append(name_and_sid["name"])
      self.sid_list.append(name_and_sid["sid"])
      
      self.mel_list.append(np.load(dat_path + "_mel.npy", mmap_mode="r"))
      self.mfcc_list.append(np.load(dat_path + "_mfcc.npy", mmap_mode="r"))
      self.chroma_list.append(np.load(dat_path + "_chroma.npy", mmap_mode="r"))
      self.vad_list.append(np.load(dat_path + "_vad.npy", mmap_mode="r"))
      self.tag_list.append(np.load(dat_path + "_tag.npy", mmap_mode="r"))
    
      # validation
      assert len(self.name_list[-1]) == len(self.sid_list[-1])
      assert len(self.name_list[-1]) == len(self.mel_list[-1])
      assert len(self.name_list[-1]) == len(self.mfcc_list[-1])
      assert len(self.name_list[-1]) == len(self.chroma_list[-1])
      assert len(self.name_list[-1]) == len(self.vad_list[-1])
      assert len(self.name_list[-1]) == len(self.tag_list[-1])
      
      # append npz idx and data idx
      self.dat_idx += [didx for _ in range(len(self.name_list[-1]))]
      self.data_idx += [eidx for eidx in range(len(self.name_list[-1]))]
    
    # get mean and std of vad. 
    vad_total = np.concatenate(self.vad_list, axis=0)
    self.normalize_vad = normalize_vad
    self.vad_mean = torch.tensor(np.average(vad_total, axis=0), dtype=torch.float32, device=device)
    self.vad_std = torch.tensor(np.std(vad_total, axis=0), dtype=torch.float32, device=device)
    
    # get pos weight
    tag_total = np.concatenate(self.tag_list, axis=0)
    self.pos_weight = torch.tensor(np.reciprocal(np.average(tag_total, axis=0) + 1e-6) - 1.0, dtype=torch.float32, device=device)
    
    print(f"Loading dataset done!: {len(self)} items...")
  
  def __len__(self):
    """
    Return size of the dataset. 
    
    Return:
        - int: length of the dataset
    """
    return len(self.dat_idx)
  
  def __getitem__(self, idx: int):
    """
    Return data at index idx. 

    Args:
        - idx (int): index.
    
    Return:
        - tuple: list of torch tensors
    """
    # get npz idx and data idx
    didx, eidx = self.dat_idx[idx], self.data_idx[idx]
    
    # get data
    mel = torch.tensor(self.mel_list[didx][eidx], dtype=torch.float32, device=self.device)
    mfcc = torch.tensor(self.mfcc_list[didx][eidx], dtype=torch.float32, device=self.device)
    chroma = torch.tensor(self.chroma_list[didx][eidx], dtype=torch.float32, device=self.device)
    vad = torch.tensor(self.vad_list[didx][eidx], dtype=torch.float32, device=self.device)
    tag = torch.tensor(self.tag_list[didx][eidx], dtype=torch.float32, device=self.device)
    
    # randomly crop 128 in input data
    start_idx = random.randint(0, mel.shape[1] - 128)
    
    return \
      self.name_list[didx][eidx], self.sid_list[didx][eidx], \
      mel[:, start_idx:start_idx + 128].unsqueeze(0), \
      mfcc[:, start_idx:start_idx + 128].unsqueeze(0), \
      chroma[:, start_idx:start_idx + 128].unsqueeze(0), \
      (vad - self.vad_mean) / self.vad_std if self.normalize_vad else vad, \
      tag
