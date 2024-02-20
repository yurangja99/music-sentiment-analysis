from collections import namedtuple

UserValues = namedtuple("UserValues", [
  "MUSE_CSV_PATH",
  "SPOTIFY_CID",
  "SPOTIFY_SECRET",
  "EMOTION_TAG_LIST"
])

uv = UserValues(
  MUSE_CSV_PATH="./muse_v3.csv",
  SPOTIFY_CID="aba431771221492aa11591106039d865",
  SPOTIFY_SECRET="8c3e0f21aa4944988d3635bab29abc1b",
  EMOTION_TAG_LIST=[
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
)
