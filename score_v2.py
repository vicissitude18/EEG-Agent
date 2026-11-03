"""Independent annotation read, official event scorer, and selection-error audit."""
import argparse,json
from dataclasses import asdict
from paths import ROOT,dependencies,read_jsonl,save
B=dependencies()/'EEGAgentBenchmark'
from eeg_agent_bench.evaluators.event_based_f1 import score_event_based_record,metrics_from_event_counts,DEFAULT_EVENT_BASED_PARAMETERS
from eeg_agent_bench.evaluators.seizure_event import temporal_dice
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='main');args=p.parse_args();out=ROOT/'results'/args.output
    m=json.loads((ROOT/'results/manifest_v2.json').read_text());clips={f['case']:f for f in m['files']}
    gold={r['scenario_id']:r['gold']['seizure_events'] for r in read_jsonl(B/'benchmark/answer_keys/T5.jsonl')}
    rows=json.loads((out/'runs.json').read_text());cases=[];counts=[0,0,0];review_counts=[0,0,0];dice=[];screen_hits=0;gold_count=0
    for row in rows:
        f=clips[row['case']];start=f['source_start_sec'];end=start+f['duration_sec']
        truth=[[max(a,start)-start,min(b,end)-start] for a,b in gold[f['scenario_id']] if a<end and b>start]
        pred=(row.get('prediction') or {}).get('seizure_events',[]);ev=score_event_based_record(truth,pred,row['duration_sec'])
        d=temporal_dice(truth,[[e['onset_sec'],e['offset_sec']] for e in pred])
        if truth:dice.append(d)
        for i,k in enumerate(['tp','fp','fn']):counts[i]+=ev[k]
        trace_path=out/f"case_{row['case']:02d}"
        scan=json.loads((trace_path/'scan.json').read_text()) if (trace_path/'scan.json').exists() else {}
        decisions=(row.get('prediction') or {}).get('candidates',[])
        queue=[{'onset_sec':d['onset_sec'],'offset_sec':d['offset_sec']} for d in decisions if d['decision'] in ['suspected','uncertain']]
        qev=score_event_based_record(truth,queue,row['duration_sec'])
        for i,k in enumerate(['tp','fp','fn']):review_counts[i]+=qev[k]
        regions=scan.get('screening_regions',[])+[{'onset_sec':d['proposed_interval'][0],'offset_sec':d['proposed_interval'][1]} for d in decisions if d['origin']=='agent_added']
        audit=[]
        for a,b in truth:
            hit=any(min(b,r['offset_sec'])>max(a,r['onset_sec']) for r in regions);screen_hits+=hit;gold_count+=1
            matches=[{'candidate_id':d['candidate_id'],'decision':d['decision']} for d in decisions if min(b,d['review_interval'][1])>max(a,d['review_interval'][0])]
            audit.append({'reference':[a,b],'broad_or_added_region_overlaps':hit,'nearby_candidate_decisions':matches})
        cases.append({**row,'reference_events':truth,'event_metrics':ev,'review_queue_event_metrics':qev,'time_dice':d,'posthoc_error_audit':audit})
    tp,fp,fn=counts;dur=sum(r['duration_sec'] for r in rows)
    stats={k:sum(r.get('stats',{}).get(k,0) for r in rows) for k in ['model_calls','tool_dispatches','measurements_executed','cache_hits','invalid_calls','submission_rejections']}
    total={'n':len(rows),'subjects':len(set(r['subject'] for r in rows)),'duration_sec':dur,'completed':sum(r['status']=='completed' for r in rows),'tp':tp,'fp':fp,'fn':fn,**metrics_from_event_counts(tp,fp,fn,dur),
      'fp_per_hour':fp/(dur/3600),'dice_s_positive_clips':sum(dice)/len(dice) if dice else None,'candidate_screen_reference_coverage':{'hits':screen_hits,'reference_events':gold_count},
      'inclusive_review_queue':dict(zip(['tp','fp','fn'],review_counts)),'tool_statistics':stats,'candidate_decisions':{k:sum(d['decision']==k for c in cases for d in (c.get('prediction') or {}).get('candidates',[])) for k in ['suspected','rejected','uncertain']},
      'event_parameters':asdict(DEFAULT_EVENT_BASED_PARAMETERS),'official_full_benchmark_score':False,'scope':m['selection'],'cases':cases}
    save(out/'score.json',total);print(json.dumps({k:v for k,v in total.items() if k!='cases'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
