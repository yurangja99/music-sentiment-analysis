import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from model import ResidualBlock, BottleneckResidualBlock, Network

#########################################
# EDITABLE

# model hyperparameters
resume_training = False
model_version = "test"
use_mel = True
use_mfcc = False
use_chroma = False
block = BottleneckResidualBlock # BottleneckResidualBlock
num_blocks = [2, 2, 2, 2, 2, 2]

# training hyperparameters
save_epoch_freq = 10
epochs = 2000
batch_size = 32
learning_rate = 0.005

#########################################

# list of emotion tags
emotion_tag_list=[
  'acerbic', 'aggressive', 'agreeable', 'airy', 'ambitious', 'amiable', 'angry', 'angst-ridden', 
  'animated', 'anxious', 'apocalyptic', 'athletic', 'atmospheric', 'austere', 'autumnal', 
  'belligerent', 'benevolent', 'bitter', 'bittersweet', 'bleak', 'boisterous', 'bombastic', 
  'brash', 'brassy', 'bravado', 'bright', 'brittle', 'brooding', 'calm', 'campy', 'capricious', 
  'carefree', 'cathartic', 'celebratory', 'cerebral', 'cheerful', 'child-like', 'circular', 
  'clinical', 'cold', 'comic', 'complex', 'confident', 'confrontational', 'consoling', 'crunchy', 
  'cynical', 'dark', 'defiant', 'delicate', 'demonic', 'desperate', 'detached', 'devotional', 
  'difficult', 'dignified', 'distraught', 'dramatic', 'dreamy', 'driving', 'druggy', 'earnest', 
  'earthy', 'ebullient', 'eccentric', 'ecstatic', 'eerie', 'effervescent', 'elaborate', 'elegant', 
  'elegiac', 'energetic', 'enigmatic', 'epic', 'erotic', 'ethereal', 'euphoric', 'exciting', 
  'exotic', 'explosive', 'exuberant', 'feral', 'feverish', 'fierce', 'fiery', 'flashy', 'flowing', 
  'fractured', 'freewheeling', 'fun', 'funereal', 'gentle', 'giddy', 'gleeful', 'gloomy', 
  'good-natured', 'graceful', 'greasy', 'grim', 'gritty', 'gutsy', 'halloween', 'happy', 'harsh', 
  'hedonistic', 'hostile', 'humorous', 'hungry', 'hymn-like', 'hyper', 'hypnotic', 'indulgent', 
  'innocent', 'insular', 'intense', 'intimate', 'introspective', 'ironic', 'irreverent', 'jittery', 
  'jovial', 'joyous', 'kinetic', 'knotty', 'laid-back', 'languid', 'lazy', 'light', 'literate', 
  'lively', 'lonely', 'lush', 'lyrical', 'macabre', 'malevolent', 'manic', 'marching', 'martial', 
  'meandering', 'mechanical', 'meditative', 'melancholy', 'mellow', 'menacing', 'messy', 'mighty', 
  'monastic', 'monumental', 'motoric', 'mysterious', 'mystical', 'naive', 'narcotic', 'narrative', 
  'negative', 'nervous', 'nihilistic', 'noble', 'nocturnal', 'nostalgic', 'ominous', 'optimistic', 
  'opulent', 'organic', 'ornate', 'outraged', 'outrageous', 'paranoid', 'passionate', 'pastoral', 
  'peaceful', 'perky', 'philosophical', 'plaintive', 'playful', 'poignant', 'positive', 'powerful', 
  'precious', 'provocative', 'pure', 'quiet', 'quirky', 'rambunctious', 'ramshackle', 'raucous', 
  'reassuring', 'rebellious', 'reckless', 'refined', 'reflective', 'regretful', 'relaxed', 
  'reserved', 'resolute', 'restrained', 'reverent', 'rollicking', 'romantic', 'rousing', 'rowdy', 
  'rustic', 'sacred', 'sad', 'sarcastic', 'sardonic', 'satirical', 'savage', 'scary', 
  'scary music', 'searching', 'self-conscious', 'sensual', 'sentimental', 'serious', 'sexual', 
  'sexy', 'shimmering', 'silly', 'sleazy', 'slick', 'smooth', 'snide', 'soft', 'somber', 
  'soothing', 'sophisticated', 'spacey', 'sparkling', 'sparse', 'spicy', 'spiritual', 'spooky', 
  'sprawling', 'sprightly', 'springlike', 'stately', 'street-smart', 'strong', 'stylish', 
  'suffocating', 'sugary', 'summery', 'suspenseful', 'swaggering', 'sweet', 'technical', 'tender', 
  'tense', 'theatrical', 'thoughtful', 'threatening', 'thrilling', 'thuggish', 'tragic', 
  'translucent', 'transparent', 'trashy', 'trippy', 'triumphant', 'uncompromising', 'understated', 
  'unsettling', 'uplifting', 'urgent', 'virile', 'visceral', 'volatile', 'warm', 'weary', 
  'whimsical', 'wintry', 'wistful', 'witty', 'wry', 'yearning'
]

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load train and test dataset
train_dataset = CustomDataset(
  npz_path_list=[
    "generated_dataset_from_0_to_100.npz",
    #"generated_dataset_from_100_to_200.npz",
    #"generated_dataset_from_200_to_300.npz",
    #"generated_dataset_from_300_to_400.npz",
    #"generated_dataset_from_400_to_500.npz",
    #"generated_dataset_from_500_to_600.npz",
    #"generated_dataset_from_600_to_700.npz",
    #"generated_dataset_from_700_to_800.npz",
    #"generated_dataset_from_800_to_900.npz",
    #"generated_dataset_from_900_to_1000.npz",
    #"generated_dataset_from_1000_to_1100.npz",
    #"generated_dataset_from_1100_to_1200.npz",
    #"generated_dataset_from_1200_to_1233.npz",
  ],
  emotion_tag_list=emotion_tag_list,
  normalize_vad=True,
  device=device
)
test_dataset = CustomDataset(
  npz_path_list=[
    #"generated_dataset_from_0_to_100.npz",
    #"generated_dataset_from_100_to_200.npz",
    #"generated_dataset_from_200_to_300.npz",
    #"generated_dataset_from_300_to_400.npz",
    #"generated_dataset_from_400_to_500.npz",
    #"generated_dataset_from_500_to_600.npz",
    #"generated_dataset_from_600_to_700.npz",
    #"generated_dataset_from_700_to_800.npz",
    #"generated_dataset_from_800_to_900.npz",
    #"generated_dataset_from_900_to_1000.npz",
    #"generated_dataset_from_1000_to_1100.npz",
    #"generated_dataset_from_1100_to_1200.npz",
    "generated_dataset_from_1200_to_1233.npz",
  ],
  emotion_tag_list=emotion_tag_list,
  normalize_vad=True,
  device=device
)

# set train and test dataloader
train_dataloader = DataLoader(
  dataset=train_dataset,
  batch_size=batch_size,
  num_workers=0,
  shuffle=True,
  drop_last=True
)
test_dataloader = DataLoader(
  dataset=test_dataset, 
  batch_size=batch_size, 
  num_workers=0,
  shuffle=False,
  drop_last=False
)

# create or load model, optimizer, and history
if resume_training:
  # load model
  ckpt = torch.load(f"{model_version}.pt")
  model = Network(
    num_tags=len(emotion_tag_list),
    use_mel=ckpt["use_mel"],
    use_mfcc=ckpt["use_mfcc"],
    use_chroma=ckpt["use_chroma"],
    block=ResidualBlock if ckpt["block_name"] == "basic" else BottleneckResidualBlock,
    num_blocks=ckpt["num_blocks"]
  ).to(device=device)
  model.load_state_dict(ckpt["model"])
  
  # load optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
  optimizer.load_state_dict(ckpt["optimizer"])
  
  # training status and history
  start_epoch = ckpt["epoch"] + 1
  vad_loss_history = ckpt["vad_loss"]
  tag_loss_history = ckpt["tag_loss"]
  tag_precision_history = ckpt["tag_precision"]
  tag_recall_history = ckpt["tag_recall"]
else:
  # create model
  model = Network(
    num_tags=len(emotion_tag_list),
    use_mel=use_mel,
    use_mfcc=use_mfcc,
    use_chroma=use_chroma,
    block=block,
    num_blocks=num_blocks
  ).to(device=device)
  
  # create optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
  
  # training status and history
  start_epoch = 1
  vad_loss_history = []
  tag_loss_history = []
  tag_precision_history = []
  tag_recall_history = []

# training
pos_weight = torch.mean(train_dataset.tag_array, dim=0) + 1e-5
pos_weight = (torch.reciprocal(pos_weight) - 1.0) * 0.9
vad_criterion = torch.nn.HuberLoss()
tag_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
model.train()
for epoch in range(start_epoch, epochs + 1):
  print(f"Epoch {epoch}:")
  
  vad_loss_in_epoch = []
  tag_loss_in_epoch = []
  tag_precision_in_epoch = []
  tag_recall_in_epoch = []
  
  for (_, _, mel, mfcc, chroma, vad_gt, tag_gt) in tqdm(train_dataloader):
    # predict
    vad_pred, tag_pred = model(mel, mfcc, chroma)
    
    # calculate loss
    vad_loss = vad_criterion(vad_pred, vad_gt)
    tag_loss = tag_criterion(tag_pred, tag_gt)
    loss = vad_loss + tag_loss
    
    # history
    vad_loss_in_epoch.append(vad_loss.item())
    tag_loss_in_epoch.append(tag_loss.item())
    pred = torch.nn.Sigmoid()(tag_pred)
    precision = torch.sum(pred >= 0.5).item()
    recall = torch.sum(tag_gt >= 0.5).item()
    correct = torch.sum((pred >= 0.5) * (tag_gt >= 0.5)).item()
    tag_precision_in_epoch.append(correct / precision if correct > 0.0 else 0.0)
    tag_recall_in_epoch.append(correct / recall if correct > 0.0 else 0.0)
    
    # gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  # history
  vad_loss_history.append(sum(vad_loss_in_epoch) / len(vad_loss_in_epoch))
  tag_loss_history.append(sum(tag_loss_in_epoch) / len(tag_loss_in_epoch))
  tag_precision_history.append(sum(tag_precision_in_epoch) / len(tag_precision_in_epoch))
  tag_recall_history.append(sum(tag_recall_in_epoch) / len(tag_recall_in_epoch))
  
  # save model
  if epoch % save_epoch_freq == 0:
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    plt.title("VAD loss")
    plt.plot(vad_loss_history, label="VAD")
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.title("Tag loss")
    plt.plot(tag_loss_history, label="Tag")
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.title("Tag F1 Score")
    plt.plot(tag_precision_history, label="precision")
    plt.plot(tag_recall_history, label="recall")
    plt.plot([2 * p * r / (p + r) if p * r > 0.0 else 0.0 for (p, r) in zip(tag_precision_history, tag_recall_history)], label="F1 score")
    plt.ylim(bottom=0.0, top=1.0)
    plt.legend()
    plt.grid()
    plt.savefig(f"{model_version}.png")
    plt.close()
    
    torch.save({
      "model": model.state_dict(),
      "use_mel": model.use_mel,
      "use_mfcc": model.use_mfcc,
      "use_chroma": model.use_chroma,
      "block_name": "basic" if model.block == ResidualBlock else "bottleneck",
      "num_blocks": model.num_blocks,
      "optimizer": optimizer.state_dict(),
      "epoch": epoch,
      "vad_loss": vad_loss_history,
      "tag_loss": tag_loss_history,
      "tag_precision": tag_precision_history,
      "tag_recall": tag_recall_history
    }, f"{model_version}.pt")

# testing
vad_loss_test = []
tag_loss_test = []
tag_precision_test = []
tag_recall_test = []
model.eval()
for (_, _, mel, mfcc, chroma, vad_gt, tag_gt) in tqdm(test_dataloader):
  # predict
  vad_pred, tag_pred = model(mel, mfcc, chroma)
  
  # calculate loss
  vad_loss = vad_criterion(vad_pred, vad_gt)
  tag_loss = tag_criterion(tag_pred, tag_gt)
  
  # history
  vad_loss_test.append(vad_loss.item())
  tag_loss_test.append(tag_loss.item())
  pred = torch.nn.Sigmoid()(tag_pred)
  precision = torch.sum(pred >= 0.5).item()
  recall = torch.sum(tag_gt >= 0.5).item()
  correct = torch.sum((pred >= 0.5) * (tag_gt >= 0.5)).item()
  tag_precision_test.append(correct / precision if correct > 0.0 else 0.0)
  tag_recall_test.append(correct / recall if correct > 0.0 else 0.0)

# print test result
tag_precision = sum(tag_precision_test) / len(tag_precision_test)
tag_recall = sum(tag_recall_test) / len(tag_recall_test)
print("Test result:")
print(f"- VAD loss: {sum(vad_loss_test) / len(vad_loss_test)}")
print(f"- Tag loss: {sum(tag_loss_test) / len(tag_loss_test)}")
print(f"- Tag precision: {tag_precision}")
print(f"- Tag precision: {tag_recall}")
print(f"- Tag F1 Score: {2 * tag_precision * tag_recall / (tag_precision + tag_recall) if tag_precision * tag_recall > 0.0 else 0.0}")

# print example
name, sid, mel, mfcc, chroma, vad_gt, tag_gt = test_dataset[0]
mel = mel.unsqueeze(0)
mfcc = mfcc.unsqueeze(0)
chroma = chroma.unsqueeze(0)
vad_pred, tag_pred = model(mel, mfcc, chroma)
vad_pred, tag_pred = vad_pred[0], tag_pred[0]
pred = torch.nn.Sigmoid()(tag_pred)
precision = torch.sum(pred >= 0.5).item()
recall = torch.sum(tag_gt >= 0.5).item()
correct = torch.sum((pred >= 0.5) * (tag_gt >= 0.5)).item()
p = correct / precision if correct > 0.0 else 0.0
r = correct / recall if correct > 0.0 else 0.0
print("Example:")
print(f"- Name: {name}")
print(f"- SID: {sid}")
print(f"- VAD pred: {vad_pred}")
print(f"- VAD gt: {vad_gt}")
print(f"- Tag pred example: {pred[:10]}")
print(f"- Tag gt example: {tag_gt[:10]}")
print(f"- Tag pred: {[emotion_tag_list[idx] for idx in torch.where(pred >= 0.5)[0].tolist()]}")
print(f"- Tag gt: {[emotion_tag_list[idx] for idx in torch.where(tag_gt >= 0.5)[0].tolist()]}")
print(f"- Tag precision: {p}")
print(f"- Tag recall: {r}")
print(f"- F1 score: {2 * p * r / (p + r) if p * r > 0.0 else 0.0}")
