import argparse
from checkpoints import load_model, pool_mha, save_model

def main():
    p=argparse.ArgumentParser(description='Convert lab MHA (including legacy NeoX) to GQA/MQA')
    p.add_argument('--source',required=True)
    p.add_argument('--kv-heads',type=int,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    source=load_model(a.source)
    converted=pool_mha(source,a.kv_heads)
    save_model(converted,a.out)
    print('Saved:',a.out)
    print('Mean pooling changes the function. Evaluate and continue training.')
if __name__=='__main__': main()
