"""One bounded recovery pass for ALL numeric-supported uncertain outputs.

No answer key / score file is read. The previous guarded run is preserved.
The same local Qwen and inherited EEGAgent loop re-assess the measured interval.
"""
import argparse,copy,hashlib,json,shutil,time
from pathlib import Path
from types import SimpleNamespace
from run_v2 import ROOT,EvidenceAgent,LocalQwen,save,SYSTEM,function_register

class SavedEvidenceAgent(EvidenceAgent):
    """Reuses the inherited loop/assessment contract with persisted measurements.
    No signal reload, no repeated feature computation, no new measurement tools.
    """
    def __init__(self,backend,case_dir,candidate,evidence):
        self.config={'enable_rag':False};self.backend=backend;self.case_dir=case_dir
        self.trace=[];self.decisions=[];self.phase='assess_candidate';self.pending=[];self.observations=[];self.final_prediction=None
        self.stats={'model_calls':0,'tool_dispatches':0,'measurements_executed':0,'cache_hits':0,'invalid_calls':0,'submission_rejections':0}
        self.cache={};self.index=0;self.last_error='';self.guarded=True
        self.engine=SimpleNamespace(duration=1200.,candidates=[candidate],reviews={candidate['candidate_id']:evidence})
        self.messages=[{'role':'system','content':SYSTEM}]
        function_register.functions.clear();function_register.metadata.clear()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='fixed');parser.add_argument('--output',default='final');args=parser.parse_args()
    source=ROOT/'results'/args.source;out=ROOT/'results'/args.output
    assert not out.exists(),'Refined output exists; preserve it before another run'
    rows=json.loads((source/'runs.json').read_text())
    eligible=[]
    for i,row in enumerate(rows):
        for j,d in enumerate(row['prediction']['candidates']):
            if d['decision']=='uncertain' and d['measurement_support']['meets_minimum_measurement_support']:eligible.append((i,j))
    print('BOUNDARY_REVIEW_ELIGIBLE',[(rows[i]['case'],rows[i]['prediction']['candidates'][j]['candidate_id']) for i,j in eligible],flush=True)
    shutil.copytree(source,out);env=json.loads((out/'environment.json').read_text());attempts=[]
    backend=None;started=time.time()
    if eligible:
        path=json.loads((ROOT.parent/'eeg-agent-demo/results/qwen_download.json').read_text())['snapshot_path'];backend=LocalQwen(path)
    repair_seconds=0.
    for i,j in eligible:
        row=rows[i];d=copy.deepcopy(row['prediction']['candidates'][j]);cid=d['candidate_id'];t=time.time()
        folder=out/f"case_{row['case']:02d}";e=json.loads((folder/d['evidence_file']).read_text());repair_dir=out/'boundary_reviews'/f"case_{row['case']:02d}_{cid}"
        candidate={'candidate_id':cid,'onset_sec':d['proposed_interval'][0],'offset_sec':d['proposed_interval'][1],'origin':d['origin']}
        agent=SavedEvidenceAgent(backend,repair_dir,candidate,e)
        agent.question='One local boundary recovery review using already measured EEG evidence. This region has sustained numerical support but remains uncertain; its broad interval may include background. Reassess critically: either choose a tighter interval supported for >=50% of its duration and >=10 seconds, or keep rejected/uncertain. The support bins are measurement changes, NOT seizure annotations. Do not promote a physiological state change or artifact just because a threshold passes. Compare local time, spectral evolution and spatial evidence; use only the supplied evidence IDs. Give actual measured numerical reasons. No new broad scan or additional candidates.'
        agent.run(agent.question,max_rounds=4)
        record={'case':row['case'],'candidate_id':cid,'status':'completed' if agent.final_prediction is not None else 'unresolved','trace':str((repair_dir/'trace.json').relative_to(out)),'stats':agent.stats}
        if agent.final_prediction is not None:
            new=agent.final_prediction['candidates'][0];new['pre_refinement_assessment']=d;new['boundary_review_trace']=record['trace']
            row['prediction']['candidates'][j]=new;record['decision']=new['decision'];record['interval']=[new['onset_sec'],new['offset_sec']]
        elapsed=time.time()-t;repair_seconds+=elapsed;row['elapsed_seconds']+=elapsed
        for k,v in agent.stats.items():row['stats'][k]=row['stats'].get(k,0)+v
        record['elapsed_seconds']=elapsed;attempts.append(record);save(out/'boundary_review_log.json',attempts)
        print('BOUNDARY_REVIEW_DONE',json.dumps(record),flush=True)
    for row in rows:
        ds=row['prediction']['candidates'];row['prediction']['seizure_events']=[{'onset_sec':d['onset_sec'],'offset_sec':d['offset_sec'],'candidate_id':d['candidate_id'],'evidence_file':d['evidence_file']} for d in ds if d['decision']=='suspected']
        row['prediction']['uncertain_candidates']=[d['candidate_id'] for d in ds if d['decision']=='uncertain']
        folder=out/f"case_{row['case']:02d}";save(folder/'decisions.json',ds);save(folder/'prediction.json',row['prediction'])
    save(out/'runs.json',rows)
    load=backend.load_seconds if backend else 0
    env.update(parent_run=args.source,boundary_review_count=len(eligible),boundary_review_analysis_seconds=repair_seconds,boundary_review_model_load_seconds=load,
      analysis_seconds=env['analysis_seconds']+repair_seconds,model_load_seconds=env['model_load_seconds']+load,
      peak_torch_memory_bytes=max(env['peak_torch_memory_bytes'],backend.torch.cuda.max_memory_allocated() if backend else 0))
    env['code_sha256']['refine_v2.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();save(out/'environment.json',env)
    save(out/'lineage.json',{'source':'results/'+args.source,'selection':'All numeric-supported uncertain candidates; no reference annotations or scores read','original_case_trace':'Each case trace.json remains the source pass. New boundary review traces are linked from decisions and boundary_reviews/','resource_accounting':'environment totals combine source analysis and boundary recovery; model cold loads are included, not earlier debugging runs.'})
    print('REFINEMENT_COMPLETE',len(attempts),flush=True)
if __name__=='__main__':main()
