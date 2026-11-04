"""Download Tiny Shakespeare using the Python standard library."""
import hashlib
from pathlib import Path
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
path = Path('input.txt')
if path.exists():
    print('Keeping existing input.txt')
else:
    with urlopen(url, timeout=60) as response:
        data = response.read()
    text = data.decode('utf-8')
    if len(text) < 10000 or 'ROMEO' not in text:
        raise RuntimeError('Unexpected dataset response')
    path.write_bytes(data)
print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
