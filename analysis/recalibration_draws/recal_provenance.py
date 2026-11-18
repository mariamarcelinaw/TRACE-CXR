import hashlib,json
from pathlib import Path
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest_json(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
