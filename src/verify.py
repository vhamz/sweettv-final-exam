"""Independent brute-force interval oracle, dense ALS equations and AP checks."""
import csv,json,collections,time,sys,hashlib
import numpy as np
from scipy import sparse
from prepare import ROOT,DATA,OUT,timestamp,interval_matches
from model import als,ap5

def main():
 t=time.perf_counter();rng=np.random.default_rng(20261024)
 selected=set(rng.choice(5821987,size=240,replace=False).tolist());views=[]
 for i,r in enumerate(csv.DictReader((DATA/'dataset11-30.csv').open())):
  if i in selected:views.append((r['user_id'],int(r['vsetv_id']),timestamp(r['start_time']),timestamp(r['stop_time'])))
 channels=set(v[1] for v in views);stats={};samples=0
 for period,left,right in [('11-20',timestamp('2020-03-08 23:00:00'),timestamp('2020-05-18 00:00:00')),('21-30',timestamp('2020-05-18 00:00:00'),timestamp('2020-07-27 00:00:00'))]:
  grid=collections.defaultdict(list)
  for r in csv.DictReader((DATA/f'export_arh_{period}-final.csv').open()):
   channel=int(r['channel_id']);item=int(r['tv_show_id'])
   if channel in channels and item:
    start=timestamp(r['start_time'],True);end=start+int(r['duration']);grid[channel].append((start,end,item))
  for user,channel,a,b in views:
   a=max(a,left);b=min(b,right)
   if b<=a:continue
   airings=sorted(grid[channel]);entry=(airings,[v[0] for v in airings],np.maximum.accumulate([v[1] for v in airings]).tolist() if airings else [])
   actual=sorted(interval_matches(entry,a,b));expected=sorted((s,e,i,min(b,e)-max(a,s)) for s,e,i in airings if min(b,e)-max(a,s)>=120)
   assert actual==expected,(period,user,a,b);samples+=1
  limits=[];pairs=0
  name='train' if period=='11-20' else 'validation'
  for r in csv.DictReader((OUT/f'interactions_{name}.csv').open()):
   assert int(r['tv_show_id'])>0 and int(r['overlap_seconds'])>=120 and int(r['events'])>0
   assert left<int(r['last_time'])<=right
   pairs+=1
  stats[name+'_pairs_checked']=pairs
 assert abs(ap5([1,9,2,8,7],{1,2})-5/6)<1e-12
 assert ap5([1,2,3,4,5],set())==0
 # Compare implicit ALS with an independently constructed dense normal equation.
 C=np.array([[3.,0,1],[0,2,1]]);config=dict(factors=2,alpha=4.,reg=.2,epochs=1)
 X,Y=als(sparse.csr_matrix(C),**config)
 rr=np.random.default_rng(20261024);X0=rr.normal(0,.01,(2,2));Y0=rr.normal(0,.01,(3,2))
 P=(C>0).astype(float);confidence=1+4*np.log1p(C);Xd=[]
 for u in range(2):Xd.append(np.linalg.solve(Y0.T@np.diag(confidence[u])@Y0+.2*np.eye(2),Y0.T@(confidence[u]*P[u])))
 Xd=np.array(Xd);Yd=[]
 for i in range(3):Yd.append(np.linalg.solve(Xd.T@np.diag(confidence[:,i])@Xd+.2*np.eye(2),Xd.T@(confidence[:,i]*P[:,i])))
 Yd=np.array(Yd);assert np.max(abs(X-Xd))<1e-12 and np.max(abs(Y-Yd))<1e-12
 result=json.loads((OUT/'validation_results.json').read_text());manual=[]
 for ex in result['examples']:
  truth=set(ex['actual_validation'])
  for method,rows in ex['lists'].items():
   hits=[int(r['id'] in truth) for r in rows];cum=np.cumsum(hits);terms=[float(h*cum[k]/(k+1)) for k,h in enumerate(hits)];expected=sum(terms)/min(5,len(truth))
   got=ap5([r['id'] for r in rows],truth);assert abs(got-expected)<1e-12
   manual.append(dict(user=ex['user_id'],method=method,hits=hits,precision_terms=terms,denominator=min(5,len(truth)),ap5=got))
 bootstrap={}
 for method in ['content','mf','history','fusion']:
  delta=np.array(result['per_user'][method])-result['per_user']['popularity'];rr=np.random.default_rng(20261024);means=np.array([rr.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
  bootstrap[method]=dict(delta=float(delta.mean()),ci95=np.percentile(means,[2.5,97.5]).tolist())
 stats.update(interval_views_checked=samples,interval_matches_equal=True,implicit_als_max_parameter_error=float(max(np.max(abs(X-Xd)),np.max(abs(Y-Yd)))),manual_ap=manual,paired_bootstrap_vs_popularity=bootstrap,seconds=time.perf_counter()-t)
 path=ROOT/'submission_final.csv'
 if path.exists():
  rows=list(csv.DictReader(path.open()));template=list(csv.DictReader((DATA/'submission.csv').open()));candidate=set(json.loads((OUT/'metadata_31-42.json').read_text()))
  assert len(rows)==1260 and [r['user_id'] for r in rows]==[r['user_id'] for r in template]
  for r in rows:
   ids=r['tv_show_id'].split();assert len(ids)==5 and len(set(ids))==5 and all(i in candidate and i!='0' for i in ids)
  stats['submission_verified']=True;stats['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 (OUT/'verification.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
