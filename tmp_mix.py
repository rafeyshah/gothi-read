import json
from collections import Counter

def mix_stats(path):
    total=multi=0
    font_counts=Counter()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            fonts=r.get('gt_fonts')
            if not fonts: continue
            total+=1
            font_counts.update(set(fonts))
            if len(set(fonts))>1:
                multi+=1
    return total,multi,font_counts

for name in ['configs/train_align.jsonl','configs/valid_align.jsonl']:
    tot,multi,fc=mix_stats(name)
    print(name, 'lines',tot, 'multi-font lines', multi, 'ratio', f'{multi/tot:.4f}')
