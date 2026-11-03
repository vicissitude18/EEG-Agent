"""Integration preflight with a clearly labelled stub; never a model result."""
import json,copy,tempfile
from pathlib import Path
from run_v2 import ROOT,BENCH,EvidenceAgent,EEGAgent,Scenario,read_jsonl
class Stub:
    def generate(self,messages,tools):
        fn=tools[0]['function'];name=fn['name'];args={}
        context=json.loads(messages[1]['content'].split('\n',1)[1])
        if name=='propose_additional_regions':args={'onsets_sec':[],'offsets_sec':[],'reasons':[]}
        if name=='measure_candidate':args={'candidate_id':context['current_candidate']['candidate_id']}
        if name=='assess_candidate':
            e=context['local_evidence'];a,b=e['proposed_interval'];cid=e['candidate_id']
            args={'candidate_id':cid,'decision':'rejected','onset_sec':a,'offset_sec':b,'evolution_supported':False,'context_contrast':False,'artifact_suspected':False,'alternative_explanation':'Preflight stub, not medical inference'}
            for family in ['temporal','spectral','cross_channel']:args.update({family+'_evidence_id':e['evidence_ids'][family],family+'_reason':'Preflight stub testing provenance only; not a medical interpretation.'})
        raw=json.dumps({'name':name,'arguments':args})
        return '<tool_call>'+raw+'</tool_call>',[{'name':name,'arguments':args}],{'input_tokens':0,'output_tokens':0,'generation_seconds':0}
def main():
    from schema_utils import check_schema_isolation
    check_schema_isolation()
    assert EvidenceAgent.run is EEGAgent.run
    manifest=json.loads((ROOT/'results/manifest_v2.json').read_text());f=manifest['files'][0]
    s=copy.deepcopy(read_jsonl(BENCH/'benchmark/scenarios/T5.jsonl')[0]);s['scenario_id']='demo';s['gold_private']={}
    r=s['records'][0];r.update(record_id='record',data_path=f['data_path']);r['metadata'].update(duration_sec=1200,channel_names=f['channel_names'])
    s['access']['allowed_tools']=['scan_recording','propose_additional_regions','measure_candidate']
    with tempfile.TemporaryDirectory() as tmp:
        a=EvidenceAgent(Scenario.from_dict(s),Stub(),Path(tmp));a.run(a.question,max_rounds=24)
        assert a.final_prediction is not None and a.stats['invalid_calls']==0
        assert len(a.engine.scan_output['rows'])==120 and a.engine.scan_output['rows'][-1][1]==1200
        assert len(a.engine.channels)<len(f['channel_names'])
        assert all(all(k in d for k in ['temporal_evidence_id','spectral_evidence_id','cross_channel_evidence_id']) for d in a.decisions)
        before=a.stats['measurements_executed'];a.wrap('scan_recording')();assert a.stats['cache_hits']==1 and a.stats['measurements_executed']==before
        if a.decisions:
            a.index=0;a.phase='assess_candidate';bad=copy.deepcopy(a.decisions[0]);bad['temporal_evidence_id']='invented'
            try:a.assess(bad)
            except ValueError:pass
            else:raise AssertionError('Invented evidence accepted')
        try:a.engine.add_regions([{'onset_sec':-1,'offset_sec':10}])
        except ValueError:pass
        else:raise AssertionError('Out-of-range interval accepted')
    print('PREFLIGHT_PASS: inherited EEGAgent loop and dispatcher; contiguous 20min scan; dedup; per-candidate 3-family evidence; cache; invalid evidence and interval rejection. Stub results discarded.')
if __name__=='__main__':main()
