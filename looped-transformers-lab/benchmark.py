"""Optional local Ollama benchmark. No generated code is executed."""
import argparse, json, time, urllib.request, platform
from pathlib import Path

def generate(model, prompt, ctx, limit, think):
    payload = {'model': model, 'prompt': prompt, 'stream': True, 'options': {'num_ctx': ctx, 'num_predict': limit, 'temperature': 0}, 'keep_alive': 0}
    if think is not None:
        payload['think'] = think
    req = urllib.request.Request('http://localhost:11434/api/generate', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    first = None
    text = []
    reasoning = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            obj = json.loads(line)
            content = obj.get('response', '')
            thought = obj.get('thinking', '')
            if first is None and (content or thought):
                first = time.perf_counter() - start
            text.append(content)
            reasoning.append(thought)
            if obj.get('done'):
                break
    if not obj.get('done'):
        raise RuntimeError('incomplete stream')
    return {'model': model, 'wall_seconds': time.perf_counter() - start, 'ttft_seconds': first, 'generated_tokens': obj.get('eval_count'), 'decode_tps': obj.get('eval_count', 0) / (obj.get('eval_duration', 0) / 1000000000.0) if obj.get('eval_duration') else None, 'prompt_tokens': obj.get('prompt_eval_count'), 'prompt_seconds': obj.get('prompt_eval_duration', 0) / 1000000000.0, 'load_seconds': obj.get('load_duration', 0) / 1000000000.0, 'done_reason': obj.get('done_reason'), 'response': ''.join(text), 'thinking': ''.join(reasoning)}
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['inkling-2-mini', 'qwen3.8:27b'])
    p.add_argument('--prompt', default='Implement an O(1) LRU cache in Python and explain how to test it.')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--ctx', type=int, default=8192)
    p.add_argument('--limit', type=int, default=2048)
    p.add_argument('--think', choices=['default', 'true', 'false'], default='default')
    p.add_argument('--out', default='results/ollama.json')
    a = p.parse_args()
    rows = []
    with urllib.request.urlopen('http://localhost:11434/api/version') as r:
        version = json.load(r)
    with urllib.request.urlopen('http://localhost:11434/api/tags') as r:
        tags = json.load(r)
    for i in range(a.repeats):
        for m in a.models if i % 2 == 0 else a.models[::-1]:
            row = generate(m, a.prompt, a.ctx, a.limit, None if a.think == 'default' else a.think == 'true')
            row['repeat'] = i
            rows.append(row)
            print(m, row['wall_seconds'], flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps({'settings': vars(a), 'platform': platform.platform(), 'ollama': version, 'installed_models': tags, 'runs': rows}, indent=2))
