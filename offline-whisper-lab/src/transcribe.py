#!/usr/bin/env python3
"""Whisper CLI file transcription, timestamps, provenance and real-time factor."""
from __future__ import annotations
import argparse,csv,json,platform,subprocess,time,wave
from pathlib import Path
from common import audio_duration,hash_file,parse_segments,write_srt

def run(cli:Path,model:Path,wav:Path,outdir:Path,language='en',threads=4,repeat=1,reference:Path|None=None):
    if not cli.exists():raise FileNotFoundError(f'whisper-cli not found: {cli}')
    if not model.exists():raise FileNotFoundError(f'model not found: {model}')
    if not wav.exists():raise FileNotFoundError(f'WAV not found: {wav}')
    with wave.open(str(wav),'rb') as w:
        if (w.getnchannels(),w.getframerate(),w.getsampwidth())!=(1,16000,2):
            raise ValueError('Expect 16-kHz mono 16-bit PCM WAV; use FFmpeg conversion instructions')
    outdir.mkdir(parents=True,exist_ok=True)
    duration=audio_duration(wav)
    if duration<=0:raise ValueError('Empty audio')
    rows=[]
    for i in range(repeat):
        stem=outdir/f'trial_{i+1:02d}'
        cmd=[str(cli),'-m',str(model),'-f',str(wav),'-l',language,'-t',str(threads),'-oj','-of',str(stem)]
        start=time.perf_counter()
        proc=subprocess.run(cmd,text=True,capture_output=True)
        elapsed=time.perf_counter()-start
        (outdir/f'trial_{i+1:02d}.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(f'whisper-cli failed (trial {i+1}, exit {proc.returncode}); see .log')
        json_path=Path(str(stem)+'.json')
        if not json_path.exists():raise RuntimeError('whisper-cli produced no JSON output; check installed flags/version')
        raw=json.loads(json_path.read_text(encoding='utf-8'))
        segments=parse_segments(raw)
        transcript=' '.join(s['text'] for s in segments).strip()
        (outdir/f'trial_{i+1:02d}.txt').write_text(transcript+'\n',encoding='utf-8')
        write_srt(segments,outdir/f'trial_{i+1:02d}.srt')
        row={'trial':i+1,'audio_seconds':round(duration,4),'wall_seconds':round(elapsed,4),
             'rtf':round(elapsed/duration,4),'speed_x_realtime':round(duration/elapsed,4),
             'segments':len(segments),'transcript':transcript}
        if reference:
            from common import word_errors
            row.update(word_errors(reference.read_text(encoding='utf-8'),transcript))
        rows.append(row)
        print(f'Trial {i+1}: wall={elapsed:.2f}s audio={duration:.2f}s RTF={elapsed/duration:.3f} transcript={transcript[:80]!r}')
    with (outdir/'runs.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    meta={'platform':platform.platform(),'model_sha256':hash_file(model),'wav_sha256':hash_file(wav),
          'model':str(model),'audio':str(wav),'language':language,'threads':threads,'cli':str(cli),
          'benchmark_definition':'wall time includes model loading, file decoding, inference, and output writing',
          'rtf_definition':'wall_seconds / audio_seconds, below 1 indicates faster-than-real-time batch completion'}
    (outdir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return rows

def main():
    p=argparse.ArgumentParser(description='Local whisper.cpp transcription with RTF and optional WER.')
    p.add_argument('--cli',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/first'))
    p.add_argument('--reference',type=Path);p.add_argument('--language',default='en');p.add_argument('--threads',type=int,default=4);p.add_argument('--repeat',type=int,default=3)
    a=p.parse_args()
    if a.repeat<1 or a.threads<1:p.error('repeat and threads must be >= 1')
    run(a.cli.resolve(),a.model.resolve(),a.audio.resolve(),a.output,a.language,a.threads,a.repeat,a.reference)
if __name__=='__main__':main()
