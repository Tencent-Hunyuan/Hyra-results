"""Reproduce randomized node checks against exhaustive zero-syndrome completions.
Run with PYTHONPATH=tests:. python3 scripts/v2-node-experiment.py.
"""
import itertools,random,collections,json
from cert_factory import Fixture
from v2_factory import NodeDriver
import oracle
rng=random.Random(171)
counts=collections.Counter();examples={}
for n in (5,6,7,8,9):
 for instance in range(4):
  rows=[];group={'I'*n}
  while len(group)<2**(n-1):
   g=''.join(rng.choices('IXZY',k=n))
   if g not in group and all(oracle.commutes(g,h) for h in rows):
    rows.append(g);group=oracle.closure(n,rows)
  probe=Fixture(n,rows,0,1)
  cap=4
  completions=[tuple('IXZY'.index(c) for c in p) for p in oracle.all_paulis(n) if sum(c!='I' for c in p)<=cap and all(oracle.commutes(p,g) for g in rows)]
  driver=NodeDriver(probe,'s0s1s2',cap)
  try:
   for trial in range(1200):
    lab=[rng.choice([0,4,5,6,7]) for q in range(n)]
    for q in rng.sample(range(n),rng.randrange(1,4)):lab[q]=rng.randrange(1,4)
    out=driver.query(lab)
    if 'error' in out:
     if 'logical leaf' in out['error']:continue
     raise AssertionError((n,rows,lab,out))
    rule=out.get('rule',-1);counts[rule]+=1
    if rule in (2,5,6,7):
     assert not any(all((a==b if a<4 else True if a==4 else b in (0,a-4)) for a,b in zip(lab,p)) for p in completions),(rows,lab,out)
     if rule not in examples:examples[rule]={'n':n,'rows':rows,'cap':cap,'lab':lab,'expected':out};print('example',rule,examples[rule],flush=True)
  finally:driver.close()
 print(n,dict(counts),flush=True)
print('all',dict(counts))
print(json.dumps({'examples':examples,'counts':dict(counts)},sort_keys=True))
