from collections import namedtuple

UserValues = namedtuple("UserValues", [
  "MUSE_CSV_PATH",
  "SPOTIFY_CID",
  "SPOTIFY_SECRET"
])

uv = UserValues(
  MUSE_CSV_PATH="MUSE_CSV_PATH",
  SPOTIFY_CID="SPOTIFY_CID",
  SPOTIFY_SECRET="SPOTIFY_SECRET"
)
