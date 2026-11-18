#!/usr/bin/env python3
"""Explicitly user-initiated microphone capture via FFmpeg; no background recording."""
import argparse,platform,shutil,subprocess
from pathlib import Path

def command(seconds:int,out:Path,input_device:str|None=None,backend:str|None=None):
    if not 1<=seconds<=3600: raise ValueError('Duration must be 1..3600 seconds')
    os_name=platform.system()
    backend=backend or {'Darwin':'avfoundation','Linux':'pulse','Windows':'dshow'}.get(os_name)
    if backend not in {'avfoundation','pulse','dshow'}: raise ValueError('Unsupported recorder backend')
    source=input_device or {'avfoundation':':0','pulse':'default','dshow':None}[backend]
    if not source: raise ValueError('Windows dshow needs --device "audio=Your Microphone Name"')
    return ['ffmpeg','-hide_banner','-nostdin','-loglevel','error','-y','-f',backend,'-i',source,'-t',str(seconds),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(out)]

def main():
    p=argparse.ArgumentParser(description='Record a short, explicitly initiated microphone clip locally.')
    p.add_argument('--seconds',type=int,default=20)
    p.add_argument('--output',type=Path,default=Path('data/recording.wav'))
    p.add_argument('--device',help='FFmpeg input device identifier; e.g. macOS :0, Linux default, Windows audio=Microphone Name')
    p.add_argument('--backend',choices=['avfoundation','pulse','dshow'])
    a=p.parse_args()
    if not shutil.which('ffmpeg'):p.error('ffmpeg is required; install it locally')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=command(a.seconds,a.output,a.device,a.backend)
    print('Recording starts now and stops automatically after',a.seconds,'seconds. Audio stays local.')
    try:subprocess.run(cmd,check=True)
    except subprocess.CalledProcessError as exc:raise SystemExit(f'Recording failed ({exc.returncode}). Check microphone permission/device name.')
    print('Saved:',a.output)
if __name__=='__main__':main()
