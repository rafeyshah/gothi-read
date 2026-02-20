import json
from pathlib import Path
fn=Path('configs/train_align_balanced.jsonl')
with fn.open('r',encoding='utf-8') as f:
    for i,line in enumerate(f,1):
        try:
            json.loads(line)
        except Exception as e:
            print('bad line', i, e)
            print(line[:400])
            break
