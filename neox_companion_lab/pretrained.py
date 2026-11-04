"""Optional pretrained Pythia generation and bias-only adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import device_for

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='EleutherAI/pythia-70m')
    p.add_argument('--revision',default=None)
    p.add_argument('--prompt',default='The researcher discovered that')
    p.add_argument('--tokens',type=int,default=50)
    p.add_argument('--text',default=None)
    p.add_argument('--steps',type=int,default=0)
    p.add_argument('--block',type=int,default=128)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--out',default='runs/pythia_bias')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    a=p.parse_args()
    if a.steps<0 or a.batch<1 or a.tokens<1 or a.block<2:
        p.error('Invalid numerical arguments')
    if a.steps and not a.text:
        p.error('Training requires --text')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device=device_for(a.device)
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,
                                              torch_dtype=torch.float32)
    if model.config.model_type!='gpt_neox':
        p.error('Expected a GPT-NeoX-family model')
    model.to(device)
    model.config.use_cache=False
    print('Base revision:',getattr(model.config,'_commit_hash',None))
    model.requires_grad_(False)
    for name,param in model.named_parameters():
        if name.endswith('.bias'):
            param.requires_grad_(True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    print('Trainable bias parameters:',sum(p.numel() for p in parameters))
    metrics={}
    if a.text:
        raw=Path(a.text).read_bytes()
        text=raw.decode('utf-8')
        cut=int(len(text)*0.9)
        def encode(value):
            return torch.tensor(tokenizer(value,add_special_tokens=False,
                                           verbose=False)['input_ids'])
        train_ids,val_ids=encode(text[:cut]),encode(text[cut:])
        if min(len(train_ids),len(val_ids))<a.block or a.block>model.config.max_position_embeddings:
            p.error('Split too short or block exceeds model context')
        @torch.no_grad()
        def evaluate():
            model.eval()
            total,count=0.0,0
            for start in range(0,len(val_ids)-a.block+1,a.block):
                batch=val_ids[start:start+a.block][None].to(device)
                value=model(batch,labels=batch).loss.item()
                total+=value*(a.block-1)
                count+=a.block-1
            return total/count
        metrics['before_token_nll']=evaluate()
        print('Initial held-out token NLL:',metrics['before_token_nll'])
        optimizer=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=0.0)
        model.train()
        for step in range(a.steps):
            starts=torch.randint(0,len(train_ids)-a.block+1,(a.batch,))
            batch=torch.stack([train_ids[i:i+a.block] for i in starts]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=model(batch,labels=batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.0)
            optimizer.step()
            if (step+1)%10==0:
                print('step',step+1,'loss',loss.item())
        metrics['after_token_nll']=evaluate()
        metrics['data_sha256']=hashlib.sha256(raw).hexdigest()
        print('Final held-out token NLL:',metrics['after_token_nll'])
    if a.steps:
        out=Path(a.out)
        out.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(out,safe_serialization=True)
        tokenizer.save_pretrained(out)
        metrics['arguments']=vars(a)
        metrics['base_revision']=getattr(model.config,'_commit_hash',None)
        (out/'lab_metrics.json').write_text(json.dumps(metrics,indent=2))
        print('Saved full checkpoint, not bias-only delta:',out)
    model.eval()
    inputs=tokenizer(a.prompt,return_tensors='pt').to(device)
    if inputs.input_ids.shape[1]==0 or inputs.input_ids.shape[1]+a.tokens>model.config.max_position_embeddings:
        p.error('Empty prompt or context budget exceeded')
    with torch.no_grad():
        output=model.generate(**inputs,max_new_tokens=a.tokens,do_sample=False,
                              use_cache=True,pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0],skip_special_tokens=True))

if __name__=='__main__':
    main()
