import os
import re
import torch

SFT = os.environ.get('SFT_PT', 'dataset/evaluation-records/interactive-pedes-conv-log/fixed_sft15_selector.pt')
RAVEL = os.environ.get('RAVEL_PT', 'dataset/evaluation-records/interactive-pedes-conv-log/ravel-main.pt')


def qtype(q):
    t = q.lower().strip().replace('\n', ' ')
    if re.search(r'\b(a\)|b\)|c\)|d\)|e\))', t):
        return 'multiple-choice'
    if re.match(r'^(is|are|was|were|does|do|did|has|have|had)\b', t):
        return 'yes/no'
    if any(x in t for x in ['any additional details', 'any other details', 'anything else', 'more details about the person', 'appearance or surroundings', 'appearance or location', 'distinguishing features']):
        return 'broad-open'
    if re.search(r'(describe|provide|tell me|what can you tell).*(coat|jacket|shirt|top|pants|jeans|shoes|sneakers|bag|backpack|umbrella|hair|hat|hood|accessor|item|posture|environment|surroundings)', t):
        return 'local-open'
    if re.match(r'^(what|which|where|how)\b', t):
        return 'local-open'
    return 'other'


def stats(path, name):
    x = torch.load(path, map_location='cpu')
    qids, gids = x['qids'], x['gids']
    sim = x['qfeats'].float() @ x['gfeats'].float().T
    ranks = []
    for r in range(6):
        order = sim[r].argsort(dim=1, descending=True)
        ordered = gids[order]
        rr = []
        for i in range(len(qids)):
            pos = torch.where(ordered[i] == qids[i])[0][0].item() + 1
            rr.append(pos)
        ranks.append(rr)
    print('\n###', name)
    for r in range(5):
        by = {}
        for i, item in enumerate(x['conversation_log']):
            q = item['interaction'][r]['question']
            typ = qtype(q)
            before, after = ranks[r][i], ranks[r+1][i]
            drr = 1/(after+1e-9) - 1/(before+1e-9)
            ans = item['interaction'][r].get('answer','')
            neg = bool(re.search(r"\b(no|not|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|cannot|can't|do not|does not|did not)\b", ans.lower()))
            row = by.setdefault(typ, [0,0,0,0,0,0.0,0])
            row[0] += 1
            row[1] += int(after < before)
            row[2] += int(after > before)
            row[3] += int(after == 1 and before != 1)
            row[4] += int(neg)
            row[5] += drr
            row[6] += len(ans.split())
        print('round', r+1)
        for typ,row in sorted(by.items(), key=lambda z:-z[1][0]):
            n, imp, worse, hit, neg, drr, words = row
            print(f'{typ:14s} n={n:5d} pct={n/len(qids)*100:6.2f} improve={imp/n*100:6.2f} worse={worse/n*100:6.2f} hit={hit/n*100:6.2f} neg={neg/n*100:6.2f} dRR={drr/n:.5f} ans_words={words/n:.2f}')


stats(SFT, os.environ.get('SFT_NAME', 'SFT 67.9'))
stats(RAVEL, os.environ.get('RAVEL_NAME', 'RAVEL 73.73'))
