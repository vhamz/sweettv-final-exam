"""Predeclared recency/airing-context improvement, same original hold-out."""
import csv,json,time,collections
import numpy as np
from scipy import sparse
from model import ROOT,OUT,DATA,load,matrix,top,evaluate,rank_scores,features

def run(final=False):
 start=time.perf_counter();early=load('train');later=load('validation');rows=early+later if final else early
 old=json.loads((OUT/'validation_before_recency.json').read_text());targets=([r['user_id'] for r in csv.DictReader((DATA/'submission.csv').open())] if final else old['users'])
 meta=json.loads((OUT/('metadata_31-42.json' if final else 'metadata_21-30.json')).read_text());ids=np.array(sorted(map(int,meta)));ii={int(x):i for i,x in enumerate(ids)};ui={u:i for i,u in enumerate(targets)}
 trainusers={r['user_id'] for r in rows};warm=np.array([u in trainusers for u in targets]);truthmap=collections.defaultdict(set)
 for r in later:truthmap[r['user_id']].add(int(r['tv_show_id']))
 truth=[truthmap[u] for u in targets]
 users=sorted(trainusers);items=sorted(set(int(r['tv_show_id']) for r in rows));itemindex={i:j for j,i in enumerate(items)}
 C=matrix(rows,users,items);B=C.copy();B.data[:]=1;pv=np.array(B.sum(0)).ravel();pop=np.array([pv[itemindex[x]] if x in itemindex else 0 for x in ids]);pop=np.broadcast_to(pop,(len(targets),len(ids))).copy();scores={'popularity':pop}
 trials=[]
 def fix(S):
  S+=pop/(max(pop.max(),1)*1e10);S[~warm]=pop[~warm];return S
 for half in [7,14,28]:
  H=np.zeros_like(pop);recentpop=np.zeros(len(ids));userweights=collections.defaultdict(float)
  for period,rs in enumerate([early,later] if final else [early]):
   decay=2**(-70/half) if final and period==0 else 1
   for r in rs:
    item=int(r['tv_show_id'])
    if item not in ii:continue
    value=float(r['decay'+str(half)])*decay
    userweights[(r['user_id'],item)]+=value
  for (u,item),value in userweights.items():
   if u in ui:H[ui[u],ii[item]]=np.log1p(value)
   recentpop[ii[item]]+=1-np.exp(-value)
  scores['history_'+str(half)]=fix(H);scores['popularity_'+str(half)]=fix(np.broadcast_to(recentpop,pop.shape).copy())
 # Channel affinity uses all past viewing, including unlabelled context, capped
 # at 6h per interval; only candidate-channel broadcast availability is future.
 availability=json.loads((OUT/('availability_31-42.json' if final else 'availability_21-30.json')).read_text());channels=sorted({int(c) for v in availability.values() for c in v});ci={c:i for i,c in enumerate(channels)}
 A=np.zeros((len(channels),len(ids)))
 for item,values in availability.items():
  for channel,duration in values.items():A[ci[int(channel)],ii[int(item)]]+=duration
 A/=np.maximum(A.sum(1,keepdims=True),1)
 for half in [7,14,28]:
  U=np.zeros((len(targets),len(channels)))
  for period,name in enumerate(['train','validation'] if final else ['train']):
   decay=2**(-70/half) if final and period==0 else 1
   for r in csv.DictReader((OUT/f'context_{name}.csv').open()):
    if r['user_id'] in ui and int(r['channel_id']) in ci:U[ui[r['user_id']],ci[int(r['channel_id'])]]+=float(r['decay'+str(half)])*decay
  U/=np.maximum(U.sum(1,keepdims=True),1);scores['channel_'+str(half)]=fix(U@A)
 # A small explicit grid, not Kaggle-score tuning. All trial scores retained.
 if not final:
  for method,S in scores.items():trials.append(dict(method=method,map5=evaluate(S,ids,truth)[0]))
 rank={m:rank_scores(S,pop) for m,S in scores.items()};best=-1;config=None;winning=None
 configs=[]
 for half in [7,14,28]:
  for history_weight in [.25,.5,.75]:
   configs.append({'popularity':1-history_weight,'history_'+str(half):history_weight})
   configs.append({'popularity_'+str(half):1-history_weight,'history_'+str(half):history_weight})
  for ch_weight in [.1,.25,.5]:configs.append({'popularity':(1-ch_weight)/2,'history_'+str(half):(1-ch_weight)/2,'channel_'+str(half):ch_weight})
 if final:configs=[json.loads((OUT/'recency_results.json').read_text())['selected']]
 for weights in configs:
  S=sum(w*rank[m] for m,w in weights.items());S[~warm]=pop[~warm]
  metric=None if final else evaluate(S,ids,truth)[0]
  if not final:trials.append(dict(weights=weights,map5=metric))
  if final or metric>best:best=metric or 0;config=weights;winning=S
 if not final:
  examples=[]
  for ex in old['examples']:
   u=ex['user_id'];k=targets.index(u);lists={}
   for method,S in {'recent_history':scores['history_14'],'channel':scores['channel_14'],'improved_fusion':winning}.items():
    lists[method]=[dict(id=int(ids[j]),title=meta[str(ids[j])]['tv_show_title'],category=meta[str(ids[j])]['tv_show_category'],hit=int(ids[j]) in truthmap[u]) for j in top(S[k],ids)]
   examples.append(dict(user_id=u,lists=lists))
  output=dict(selected=config,map5=best,per_user=evaluate(winning,ids,truth)[1],trials=trials,examples=examples,seconds=time.perf_counter()-start,protocol=old['protocol'],grid_size=len(configs),note='Fixed original hold-out; refinement after first Kaggle screenshot, but no hidden labels or future viewing features used.')
  (OUT/'recency_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2));print('RECENCY',best,config, 'BEST_TRIALS',sorted(trials,key=lambda r:-r['map5'])[:6],flush=True)
 else:
  with (ROOT/'submission_khamzat.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['user_id','tv_show_id'])
   for u,s in zip(targets,winning):w.writerow([u,' '.join(map(str,ids[top(s,ids)]))])
  print('FINAL_V2',config,'seconds',time.perf_counter()-start)
if __name__=='__main__':
 import sys
 run('--final' in sys.argv)
