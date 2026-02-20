import json, collections

def stats(path):
    fc=collections.Counter(); pc=collections.Counter(); tokens=lines=0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            fonts = r.get('gt_fonts') or r.get('fonts') or r.get('font_ids')
            if not fonts:
                continue
            lines+=1; tokens+=len(fonts); fc.update(fonts)
            if len(fonts)>1:
                pc.update(zip(fonts, fonts[1:]))
    return lines,tokens,fc,pc

for name in ['configs/train_align.jsonl','configs/valid_align.jsonl']:
    lines,tokens,fc,pc=stats(name)
    print(name, 'lines',lines,'tokens',tokens)
    print('fonts top5', fc.most_common(5))
    print('fonts bottom5', fc.most_common()[-5:])
    print('pairs top5', pc.most_common(5))
    print('pairs bottom5', pc.most_common()[-5:])
    print('-'*60)
