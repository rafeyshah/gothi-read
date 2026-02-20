import json
from itertools import islice
with open('configs/train_align.jsonl','r',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r.get('ok_align') is True:
            print(json.dumps(r)[:400])
            print('keys', r.keys())
            break
