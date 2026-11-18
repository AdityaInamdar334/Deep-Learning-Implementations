from __future__ import annotations
import hashlib,json,re,unicodedata,wave
from pathlib import Path

def audio_duration(path:Path)->float:
    with wave.open(str(path),'rb') as w:
        return w.getnframes()/w.getframerate()

def hash_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def timestamp_ms(ms:int)->str:
    seconds, milli=divmod(max(0,int(ms)),1000)
    minutes,secs=divmod(seconds,60)
    hours,mins=divmod(minutes,60)
    return f'{hours:02d}:{mins:02d}:{secs:02d},{milli:03d}'

def parse_segments(raw:dict)->list[dict]:
    items=raw.get('transcription',raw.get('segments',[]))
    if not isinstance(items,list): raise ValueError('whisper JSON transcription not list')
    segments=[]
    for item in items:
        offsets=item.get('offsets') or {}
        if 'from' in offsets and 'to' in offsets:
            start_ms,end_ms=int(offsets['from']),int(offsets['to'])
        else:
            # Alternate schemas may expose seconds in start/end.
            if 'start' not in item or 'end' not in item: raise ValueError('Missing segment offsets')
            start_ms,end_ms=round(float(item['start'])*1000),round(float(item['end'])*1000)
        if end_ms<start_ms: raise ValueError('Negative segment duration')
        segments.append({'start_ms':start_ms,'end_ms':end_ms,'text':str(item.get('text','')).strip()})
    return segments

def write_srt(segments:list[dict],dest:Path)->None:
    lines=[]
    for i,s in enumerate(segments,1):
        lines.append(f"{i}\n{timestamp_ms(s['start_ms'])} --> {timestamp_ms(s['end_ms'])}\n{s['text']}\n")
    dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def normalized_words(text:str)->list[str]:
    s=unicodedata.normalize('NFKC',text).casefold()
    # Explicit English-oriented evaluation normalization, retains apostrophe within words.
    s=re.sub(r"[^\w\s']",' ',s,flags=re.UNICODE)
    s=re.sub(r'\s+',' ',s)
    return s.strip().split() if s.strip() else []

def word_errors(ref:str,hyp:str)->dict:
    a,b=normalized_words(ref),normalized_words(hyp)
    n,m=len(a),len(b)
    if n==0: raise ValueError('WER undefined for empty normalized reference')
    dp=[[(0,0,0,0) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=(i,0,i,0)
    for j in range(1,m+1): dp[0][j]=(j,0,0,j)
    for i in range(1,n+1):
        for j in range(1,m+1):
            if a[i-1]==b[j-1]: dp[i][j]=dp[i-1][j-1]
            else:
                candidates=[(dp[i-1][j-1][0]+1,dp[i-1][j-1][1]+1,dp[i-1][j-1][2],dp[i-1][j-1][3]),
                    (dp[i-1][j][0]+1,dp[i-1][j][1],dp[i-1][j][2]+1,dp[i-1][j][3]),
                    (dp[i][j-1][0]+1,dp[i][j-1][1],dp[i][j-1][2],dp[i][j-1][3]+1)]
                dp[i][j]=min(candidates,key=lambda z:z[0])
    _,sub,dele,ins=dp[n][m]
    return {'reference_words':n,'hypothesis_words':m,'substitutions':sub,'deletions':dele,'insertions':ins,'wer':(sub+dele+ins)/n}
