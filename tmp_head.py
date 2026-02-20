import json
from itertools import islice
with open('configs/train_align.jsonl','r',encoding='utf-8') as f:
    for line in islice(f,5):
        print(line[:200])
        try:
            r=json.loads(line)
            print('keys', r.keys())
        except Exception as e:
            print('parse error', e)
