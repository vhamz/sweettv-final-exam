"""Own NumPy implicit ALS + metadata cosine profiles; common temporal MAP@5."""
import csv,json,time,platform,sys,re,collections,hashlib,os
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts';SEED=20261024
DATA=Path(os.environ.get('SWEETTV_DATA','/Users/kmyusupov/Downloads/sweettv-tv-program-recommender'))
def load(name):return list(csv.DictReader((OUT/f'interactions_{name}.csv').open()))
def features(m):
 d={}
 for field in ['tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3']:
  if m.get(field):d[field+'='+m[field].strip().lower()]=1.
 for field in ['director','actors']:
  for token in re.split('[,;|]',m.get(field,'')):
   if token.strip():d[field+'='+token.strip().lower()]=1.
 for year in re.findall(r'\d{4}',m.get('year_of_production',''))[:1]:d['decade='+str(int(year)//10*10)]=1.
 return d

def matrix(rows,users,items,signal='events'):
 ui={u:i for i,u in enumerate(users)};ii={v:i for i,v in enumerate(items)}
 a=[];b=[];v=[]
 for r in rows:
  item=int(r['tv_show_id']);u=r['user_id']
  if item in ii and u in ui:
   a.append(ui[u]);b.append(ii[item]);v.append(float(r[signal]))
 return sparse.csr_matrix((v,(a,b)),shape=(len(users),len(items)),dtype=np.float64)
def als(C,factors=32,alpha=20.,reg=.1,epochs=12):
 rng=np.random.default_rng(SEED);X=rng.normal(0,.01,(C.shape[0],factors));Y=rng.normal(0,.01,(C.shape[1],factors));CT=C.T.tocsr();I=np.eye(factors)*reg
 def update(A,B,M):
  gram=B.T@B+I
  for u in range(M.shape[0]):
   lo,hi=M.indptr[u:u+2];ix=M.indices[lo:hi];r=M.data[lo:hi]
   if not len(ix):A[u]=0;continue
   confidence=alpha*np.log1p(r);V=B[ix]
   A[u]=np.linalg.solve(gram+(V.T*confidence)@V,V.T@(1+confidence))
 for epoch in range(epochs):
  update(X,Y,C);update(Y,X,CT)
  if epoch%4==3:print('ALS',factors,alpha,reg,'epoch',epoch+1,flush=True)
 return X,Y

def top(scores,ids):
 # Stable exact tie-break: decreasing score then ascending numeric programme ID.
 return np.lexsort((ids,-np.asarray(scores)))[:5]
def ap5(pred,truth):
 if not truth:return 0.
 hit=0;score=0
 for rank,item in enumerate(pred,1):
  if int(item) in truth:hit+=1;score+=hit/rank
 return score/min(5,len(truth))
def evaluate(S,ids,truth):
 per=[ap5(ids[top(s,ids)],t) for s,t in zip(S,truth)]
 return float(np.mean(per)),per

def rank_scores(S,pop):
 # Reciprocal-rank scores. Stable ties retain ascending candidate IDs.
 order=np.argsort(-S,axis=1,kind='stable');ranks=np.empty_like(order)
 np.put_along_axis(ranks,order,np.arange(S.shape[1])[None,:],axis=1)
 return 1/(60+ranks+1)

def run(final=False):
 t=time.perf_counter();train=load('train');val=load('validation')
 meta0=json.loads((OUT/'features_11-20.json').read_text());meta1=json.loads((OUT/'features_21-30.json').read_text());meta2=json.loads((OUT/'metadata_31-42.json').read_text())
 candidate_validation=json.loads((OUT/'metadata_21-30.json').read_text())
 # Metadata dictionaries are extracted only from rows before each fitting cutoff.
 if final:train=train+val;meta={**meta0,**meta1};candidate_meta=meta2;selected=json.loads((OUT/'validation_results.json').read_text())['selected']
 else:meta=meta0;candidate_meta=candidate_validation
 ids=np.array(sorted(map(int,candidate_meta)));items=sorted(set(int(r['tv_show_id']) for r in train));users=sorted(set(r['user_id'] for r in train));ui={u:i for i,u in enumerate(users)};ii={v:i for i,v in enumerate(items)}
 testusers=[r['user_id'] for r in csv.DictReader((DATA/'submission.csv').open())]
 # Evaluate all target-template users with >=1 labelled hold-out interaction;
 # cold-start training users remain included. Also retain all-user sensitivity.
 truthmap=collections.defaultdict(set);candidate_set=candidate_meta_int(candidate_meta)
 if not final:
  for r in val:
   if int(r['tv_show_id']) in candidate_set:truthmap[r['user_id']].add(int(r['tv_show_id']))
 targets=testusers if final else [u for u in testusers if truthmap[u]]
 truth=[truthmap[u] for u in targets]
 C=matrix(train,users,items);binary=C.copy();binary.data[:]=1
 poptrained=np.asarray(binary.sum(axis=0)).ravel();pop=np.array([poptrained[ii[x]] if x in ii else 0 for x in ids]);popmat=np.broadcast_to(pop,(len(targets),len(ids))).astype(float).copy()
 known=np.array([ii.get(x,-1) for x in ids]);has=known>=0
 targetix=np.array([ui.get(u,-1) for u in targets]);warm=targetix>=0
 scores={'popularity':popmat};timings={};trials=[]
 def fallback(S):
  S[~warm]=popmat[~warm]
  return S
 # Content: only training metadata vocab/IDF; future schedule supplies IDs only.
 start=time.perf_counter();dv=DictVectorizer();B=dv.fit_transform([features(meta.get(str(x),{})) for x in items]);tfidf=TfidfTransformer(sublinear_tf=True);B=tfidf.fit_transform(B)
 Q=tfidf.transform(dv.transform([features(meta.get(str(x),{})) for x in ids]));W=C.copy();W.data=np.log1p(W.data);profiles=normalize(W@B)
 content=np.zeros_like(popmat);content[warm]=(profiles[targetix[warm]]@Q.T).toarray();content=fallback(content)
 # Tiny tie-break uses training popularity only; not a probability.
 content+=popmat/(max(pop.max(),1)*1e10);scores['content']=content;timings['content_seconds']=time.perf_counter()-start
 history=np.zeros_like(popmat)
 for row,u in enumerate(targets):
  if u in ui:
   source=C.getrow(ui[u]);lookup=dict(zip(source.indices,source.data));history[row]=[np.log1p(lookup.get(ii.get(x,-1),0)) for x in ids]
 history=fallback(history);history+=popmat/(max(pop.max(),1)*1e10);scores['history']=history
 settings=[selected['als_config']] if final else [dict(factors=32,alpha=20.,reg=.1,epochs=12),dict(factors=64,alpha=20.,reg=1.,epochs=12)]
 best=-1;best_config=None
 for config in settings:
  start=time.perf_counter();X,Y=als(C,**config);S=np.zeros_like(popmat);S[np.ix_(warm,has)]=X[targetix[warm]]@Y[known[has]].T;S[:,~has]=-1e20;S=fallback(S)
  metric=None if final else evaluate(S,ids,truth)[0];elapsed=time.perf_counter()-start;trials.append(dict(config=config,map5=metric,seconds=elapsed));print('TRIAL',trials[-1],flush=True)
  if final or metric>best:best=metric if metric is not None else 0;best_config=config;scores['mf']=S;timings['mf_seconds']=elapsed
 metrics={};peruser={}
 if not final:
  for name,S in scores.items():metrics[name],peruser[name]=evaluate(S,ids,truth)
 # Only rank fusion weights are selected; each component was evaluated alone.
 # Rank arrays use popularity as deterministic tie-break before conversion.
 component={name:rank_scores(S,pop) for name,S in scores.items()}
 fusion_configs=[selected['fusion_weights']] if final else [dict(mf=m,history=h,content=c,popularity=p) for m,h,c,p in [(1,0,0,0),(0,1,0,0),(.5,.5,0,0),(.25,.75,0,0),(.75,.25,0,0),(.45,.45,.1,0),(.2,.7,.1,0),(.4,.4,.1,.1),(.1,0,0,.9),(.25,0,0,.75),(.5,0,0,.5),(0,.1,0,.9),(0,.25,0,.75),(0,.5,0,.5),(.1,.1,0,.8),(.2,.3,0,.5),(0,0,.1,.9)]]
 fusion_best=-1;fusion_weights=None
 for weights in fusion_configs:
  S=sum(weight*component[name] for name,weight in weights.items());S=fallback(S)
  metric=None if final else evaluate(S,ids,truth)[0]
  if not final:trials.append(dict(fusion_weights=weights,map5=metric))
  if final or metric>fusion_best:fusion_best=metric or 0;fusion_weights=weights;scores['fusion']=S
 if not final:
  metrics['fusion'],peruser['fusion']=evaluate(scores['fusion'],ids,truth)
  winner=max(metrics,key=metrics.get)
  # Match train-history extremes among warm, hold-out-eligible template users.
  activity={u:float(C.getrow(ui[u]).sum()) for u in targets if u in ui}
  examples=[max(activity,key=activity.get),min(activity,key=activity.get)]
  details=[]
  for u in examples:
   row=targets.index(u);historyids=set(C.getrow(ui[u]).indices);watched=[items[i] for i in historyids]
   lists={name:[dict(id=int(ids[j]),title=candidate_meta[str(ids[j])]['tv_show_title'],category=candidate_meta[str(ids[j])]['tv_show_category'],hit=int(ids[j]) in truthmap[u],seen_in_training=int(ids[j]) in set(watched)) for j in top(S[row],ids)] for name,S in scores.items()}
   details.append(dict(user_id=u,events=activity[u],training_distinct=len(watched),validation_distinct=len(truthmap[u]),actual_validation=sorted(truthmap[u]),training_top=[dict(id=items[j],events=float(C[ui[u],j]),title=meta.get(str(items[j]),{}).get('tv_show_title',''),category=meta.get(str(items[j]),{}).get('tv_show_category','')) for j in sorted(historyids,key=lambda j:-C[ui[u],j])[:8]],lists=lists))
  output=dict(metrics=metrics,per_user=peruser,users=targets,cohort=len(targets),cold_start=int((~warm).sum()),excluded_no_truth=len(testusers)-len(targets),candidates=len(ids),candidate_new_items=int((~has).sum()),train_users=len(users),train_items=len(items),timings=timings,total_seconds=time.perf_counter()-t,trials=trials,examples=details,selected=dict(method=winner,als_config=best_config,fusion_weights=fusion_weights),environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),seed=SEED),protocol='train11-20;val21-30;template users with >=1 eligible labelled validation item; repeats allowed; AP denominator min(5,relevant); shared popularity fallback')
  (OUT/'validation_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2));print('METRICS',metrics,'SELECTED',output['selected'],flush=True)
 else:
  method=selected['method'];S=scores[method]
  with (ROOT/'submission_final.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['user_id','tv_show_id'])
   for u,s in zip(targets,S):writer.writerow([u,' '.join(map(str,ids[top(s,ids)]))])
  summary=dict(selected=selected,rows=len(targets),cold_start=int((~warm).sum()),candidates=len(ids),seconds=time.perf_counter()-t,submission_sha256=hashlib.sha256((ROOT/'submission_final.csv').read_bytes()).hexdigest())
  (OUT/'final_fit.json').write_text(json.dumps(summary,indent=2));print('FINAL',summary)
def candidate_meta_int(meta):return set(map(int,meta))
if __name__=='__main__':run('--final' in sys.argv)
