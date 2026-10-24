"""Streaming CSV read; channel interval join; no future viewing behaviour."""
import csv,json,time,datetime as dt,bisect,collections,hashlib,sys,os,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(sys.argv[1]) if len(sys.argv)>1 else Path(os.environ.get('SWEETTV_DATA','/Users/kmyusupov/Downloads/sweettv-tv-program-recommender'))
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
EPOCH=dt.datetime(1970,1,1)
def timestamp(s,schedule=False):
 if schedule: s=s[6:10]+'-'+s[3:5]+'-'+s[:2]+s[10:]
 return int((dt.datetime.fromisoformat(s)-EPOCH).total_seconds())
def read_schedule(name):
 channels=collections.defaultdict(list);meta={};featuremeta={};availability=collections.defaultdict(lambda:collections.defaultdict(float));seen=set();rows=labelled=duplicates=0;lo=10**15;hi=0
 for r in csv.DictReader((DATA/name).open(encoding='utf-8-sig')):
  rows+=1;start=timestamp(r['start_time'],True);duration=int(float(r['duration']));end=start+duration
  lo=min(lo,start);hi=max(hi,end);item=int(r['tv_show_id'])
  if not item or duration<=0:continue
  labelled+=1;key=(int(r['channel_id']),start,end,item)
  if key in seen:duplicates+=1;continue
  seen.add(key);channels[key[0]].append((start,end,item));availability[str(item)][str(key[0])]+=duration
  if str(item) not in meta:meta[str(item)]={k:r[k] for k in ['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director','actors']}
  cutoff=timestamp('2020-05-18 00:00:00') if '11-20' in name else timestamp('2020-07-27 00:00:00')
  if start<cutoff and str(item) not in featuremeta:featuremeta[str(item)]={k:r[k] for k in ['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director','actors']}
 for channel,airings in channels.items():
  airings.sort();channels[channel]=(airings,[r[0] for r in airings],np.maximum.accumulate([r[1] for r in airings]).tolist())
 (OUT/('features_'+name.split('_')[2].split('-final')[0]+'.json')).write_text(json.dumps(featuremeta,ensure_ascii=False))
 (OUT/('availability_'+name.split('_')[2].split('-final')[0]+'.json')).write_text(json.dumps(availability))
 return channels,meta,dict(rows=rows,labelled_rows=labelled,duplicate_labelled_rows=duplicates,items=len(meta),start=lo,end=hi,channels=len(channels))
def interval_matches(entry,start,end,threshold=120):
 airings,starts,prefix_ends=entry
 first=bisect.bisect_right(prefix_ends,start);last=bisect.bisect_left(starts,end)
 for k in range(first,last):
  sa,sb,item=airings[k];overlap=min(end,sb)-max(start,sa)
  if overlap>=threshold:yield sa,sb,item,overlap
def main():
 t=time.perf_counter();schedules=[];stats={}
 for period in ['11-20','21-30','31-42']:
  name=f'export_arh_{period}-final.csv';channels,meta,stat=read_schedule(name)
  schedules.append((channels,meta));stats[period]=stat
  (OUT/f'metadata_{period}.json').write_text(json.dumps(meta,ensure_ascii=False));print(period,stat,flush=True)
 cut=stats['21-30']['start'];test=stats['31-42']['start'];begin=stats['11-20']['start']
 # Half-open windows. A view crossing a boundary is clipped before joining.
 accum=[collections.defaultdict(lambda:[0,0,0,0.,0.,0.]) for _ in range(2)]
 context=[collections.defaultdict(lambda:[0.,0.,0.]) for _ in range(2)]
 counts=collections.Counter();users=[set(),set()];durations=[];examples=[]
 for r in csv.DictReader((DATA/'dataset11-30.csv').open(encoding='utf-8-sig')):
  counts['view_rows']+=1;user=r['user_id'];channel=int(r['vsetv_id']);a=timestamp(r['start_time']);b=timestamp(r['stop_time']);duration=int(r['duraton']);durations.append(duration)
  if b-a!=duration:counts['duration_mismatch']+=1
  if b<=a:counts['nonpositive_views']+=1;continue
  if a>=test:counts['future_view_rows_excluded']+=1;continue
  matched=False
  for w,(left,right) in enumerate([(begin,cut),(cut,test)]):
   start=max(a,left);end=min(b,right)
   if end<=start:continue
   users[w].add(user)
   ctx=context[w][(user,channel)];view_seconds=min(end-start,21600)
   for j,half in enumerate([7,14,28]):ctx[j]+=view_seconds*2**(-(right-end)/(86400*half))
   entry=schedules[w][0].get(channel)
   if not entry:continue
   for sa,sb,item,overlap in interval_matches(entry,start,end):
    value=accum[w][(user,item)];value[0]+=1;value[1]+=overlap;value[2]=max(value[2],min(end,sb))
    for j,half in enumerate([7,14,28]):value[3+j]+=2**(-(right-min(end,sb))/(86400*half))
    counts[f'qualified_matches_{w}']+=1;matched=True
    if len(examples)<12:examples.append(dict(window=w,user_id=user,channel=channel,view_start=a,view_stop=b,airing_start=sa,airing_end=sb,item=item,overlap=overlap))
  if matched:counts['views_with_labelled_match']+=1
 for w,name in enumerate(['train','validation']):
  with (OUT/f'interactions_{name}.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['user_id','tv_show_id','events','overlap_seconds','last_time','decay7','decay14','decay28'])
   for (user,item),values in sorted(accum[w].items()):writer.writerow([user,item,*values])
  with (OUT/f'context_{name}.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['user_id','channel_id','decay7','decay14','decay28'])
   for (user,channel),values in sorted(context[w].items()):writer.writerow([user,channel,*values])
  stats[name]=dict(users_with_log=len(users[w]),users_with_interactions=len(set(u for u,i in accum[w])),pairs=len(accum[w]))
 sample=list(csv.DictReader((DATA/'submission.csv').open()));testusers=[r['user_id'] for r in sample]
 trainusers=set(u for u,i in accum[0]);allusers=trainusers|set(u for u,i in accum[1]);stats['submission']=dict(rows=len(sample),unique_users=len(set(testusers)),cold_early=sum(u not in trainusers for u in testusers),cold_final=sum(u not in allusers for u in testusers))
 stats['views']=dict(counts);stats['view_duration_seconds']=dict(zip(['min','p25','median','p75','p95','p99','max'],map(float,np.percentile(durations,[0,25,50,75,95,99,100]))));stats['join_examples']=examples;stats['boundaries']={'training_start':begin,'validation_start':cut,'test_start':test};stats['seconds']=time.perf_counter()-t
 (OUT/'data_audit.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
