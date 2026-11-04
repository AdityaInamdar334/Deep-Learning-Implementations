from pathlib import Path
from urllib.request import urlopen
import hashlib
path=Path('input.txt')
if not path.exists():
    url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with urlopen(url, timeout=60) as response:
        data=response.read()
    if len(data)<10000 or b'ROMEO' not in data:
        raise RuntimeError('Unexpected dataset response')
    path.write_bytes(data)
print(path, 'SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
