import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import json
import math
import librosa
from sklearn import preprocessing
from tqdm import tqdm

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from user_values import uv

#########################################
# EDITABLE

DEBUG = True
PLOT = False

start_page = 0
pages_per_file = 100

#########################################

# set spotify credentials
spotify_credentials_manager = SpotifyClientCredentials(
  client_id=uv.SPOTIFY_CID,
  client_secret=uv.SPOTIFY_SECRET
)

# initialize spotify instance
spotify = spotipy.Spotify(client_credentials_manager=spotify_credentials_manager)
spotify.max_retries = 10
spotify.backoff_factor = 0.4
spotify.retries = 10

# read csv dataset and filter out null spotify id
df = pd.read_csv(uv.MUSE_CSV_PATH) \
  .dropna(subset=["spotify_id"]) \
  .drop(columns=["lastfm_url", "artist", "number_of_emotion_tags", "mbid", "genre"]) \
  .astype({
    "valence_tags": np.float32,
    "arousal_tags": np.float32,
    "dominance_tags": np.float32
  })

if DEBUG:
  print(f"Dataset of {len(df)} rows")
  print(f"First 10 rows: \n{df[:10]}")

# get features
spotify_id_list = df["spotify_id"].to_list()
end_page = math.ceil(len(spotify_id_list) / 50)
for page in range(start_page, end_page, pages_per_file):
  # generated dataset
  dataset_name = []
  dataset_sid = []
  dataset_mel = []
  dataset_mfcc = []
  dataset_chroma = []
  dataset_vad = []
  emotion_tag = []
  
  page_to = min(page + pages_per_file, end_page)
  for idx in tqdm(range(page, page_to)):
    # get tracks with preview url from spotify
    tracks = spotify.tracks(spotify_id_list[min(idx * 50, len(df)):min((idx + 1) * 50, len(df))])["tracks"]
    tracks = list(filter(lambda t: t["preview_url"] is not None, tracks))
    
    for track in tracks:
      # download preview mp3 file
      r = requests.get(track["preview_url"], stream=True)
      with open(f"temp.mp3", "wb") as f:
        for block in r.iter_content(1024):
          f.write(block)
      
      # read audio and visualize
      y, sr = librosa.load(f"temp.mp3")
      
      # Mel-spectrogram
      mel = librosa.feature.melspectrogram(y=y, sr=sr)
      mel = librosa.amplitude_to_db(mel, ref=np.max)
      mel = preprocessing.minmax_scale(mel)
      mel = np.pad(mel, ((0, 0), (0, 1280)), mode="constant", constant_values=0.0)[:, :1280] # approximately 30 seconds
      
      # MFCCs
      mfcc = librosa.feature.mfcc(y=y, sr=sr)
      mfcc = preprocessing.minmax_scale(mfcc)
      mfcc = np.pad(mfcc, ((0, 0), (0, 1280)), mode="constant", constant_values=0.0)[:, :1280] # approximately 30 seconds
      
      # Chroma frequencies
      chroma = librosa.feature.chroma_stft(y=y, sr=sr)
      chroma = preprocessing.minmax_scale(chroma)
      chroma = np.pad(chroma, ((0, 0), (0, 1280)), mode="constant", constant_values=0.0)[:, :1280] # approximately 30 seconds
      
      # generate dataset
      row = df.loc[df["spotify_id"] == track["id"]].iloc[0]
      dataset_name.append(row["track"])
      dataset_sid.append(row["spotify_id"])
      dataset_mel.append(mel)
      dataset_mfcc.append(mfcc)
      dataset_chroma.append(chroma)
      dataset_vad.append(np.array([row["valence_tags"], row["arousal_tags"], row["dominance_tags"]], dtype=np.float32))
      emotion_tag.append(np.array(json.loads(re.sub("'", '"', row["seeds"])), dtype=object))
        
      if PLOT:
        plt.figure(figsize=(24, 18))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar()
        plt.subplot(3, 1, 2)
        librosa.display.specshow(mfcc, sr=sr, x_axis="time", y_axis="linear")
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(chroma, x_axis="time", y_axis="chroma")
        plt.colorbar()
        plt.show()
        quit()
  
  # process dataset
  dataset_name = np.array(dataset_name, dtype=object)
  dataset_sid = np.array(dataset_sid, dtype=object)
  dataset_mel = np.array(dataset_mel, dtype=np.float32)
  dataset_mfcc = np.array(dataset_mfcc, dtype=np.float32)
  dataset_chroma = np.array(dataset_chroma, dtype=np.float32)
  dataset_vad = np.array(dataset_vad, dtype=np.float32)
  dataset_emotion_tag = np.array(emotion_tag, dtype=object)

  if DEBUG:
    print(f"Dataset:")
    print(f"Dataset-name: {dataset_name.shape}")
    print(f"Dataset-sid: {dataset_sid.shape}")
    print(f"Dataset-mel: {dataset_mel.shape}")
    print(f"Dataset-mfcc: {dataset_mfcc.shape}")
    print(f"Dataset-chroma: {dataset_chroma.shape}")
    print(f"Dataset-vad: {dataset_vad.shape}")
    print(f"Dataset-tag: {dataset_emotion_tag.shape}")
    print(f"Example:")
    print(f"Dataset-name: {dataset_name[0]}")
    print(f"Dataset-sid: {dataset_sid[0]}")
    print(f"Dataset-vad: {dataset_vad[0]}")
    print(f"Dataset-tag: {dataset_emotion_tag[0]}")

  # save dataset
  np.savez(
    f"generated_dataset_from_{page}_to_{page_to}.npz", 
    name=dataset_name,
    sid=dataset_sid,
    mel=dataset_mel,
    mfcc=dataset_mfcc,
    chroma=dataset_chroma,
    vad=dataset_vad,
    tag=dataset_emotion_tag
  )
  
  if DEBUG:
    print(f"Dataset saved ({page_to} / {end_page}): generated_dataset_from_{page}_to_{page_to}.npz")
