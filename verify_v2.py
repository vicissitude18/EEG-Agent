"""Audit saved evidence/decisions, code versions and policy accounting."""
import json,hashlib
from pathlib import Path
from paths import ROOT,save
from support_policy import support_summary
def main():
    manifest=json.loads((ROOT/'results/manifest_v2.json').read_text())
    assert len(manifest['files'])==6 and len({f['subject'] for f in manifest['files']})==3
    assert all(f['duration_sec']==1200 for f in manifest['files'])
    checks=[]
    for run in ['fixed','final']:
        folder=ROOT/'results'/run;env=json.loads((folder/'environment.json').read_text());score=json.loads((folder/'score.json').read_text())
        for name,digest in env['code_sha256'].items():
            path=folder/(Path(name).stem+'_snapshot.py') if (folder/(Path(name).stem+'_snapshot.py')).exists() else ROOT/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(run,name,'code hash')
        assert env['schema_alias_fix'] and score['n']==score['completed']==6
        for c in score['cases']:
            cf=folder/f"case_{c['case']:02d}";scan=json.loads((cf/'scan.json').read_text());trace=json.loads((cf/'trace.json').read_text())
            assert len(scan['rows'])==120 and scan['rows'][0][0]==0 and scan['rows'][-1][1]==1200
            assert all(scan['rows'][i][1]==scan['rows'][i+1][0] for i in range(119))
            visible=trace['system']+trace['question']+json.dumps([x['result'] for x in trace['events'] if x['kind']=='tool_result'])
            assert not any(name in visible for name in ['chb02','chb03','chb05','c1_seizure_'])
            decisions=c['prediction']['candidates'];assert len(decisions)==c['candidate_count']
            for d in decisions:
                e=json.loads((cf/d['evidence_file']).read_text());assert len(e['channels'])==4
                for family in ['temporal','spectral','cross_channel']:
                    assert d[family+'_evidence_id']==e['evidence_ids'][family]
                    assert len(d[family+'_reason'].strip())>=25 and d[family+'_reason'].strip()!=d[family+'_evidence_id']
                assert e['temporal'] and e['spectral'] and len(e['cross_channel']['pairwise_correlations'])==6
                if d['decision']=='suspected':
                    support=support_summary(e);assert support['meets_minimum_measurement_support']
                    a,b=d['onset_sec'],d['offset_sec'];covered=sum(max(0,min(b,y)-max(a,x)) for x,y in support['support_bins'])
                    assert b-a>=10 and covered>=max(10,.5*(b-a))
                if d.get('policy_downgrade_reason'):assert d['decision']=='uncertain' and d['raw_model_assessment']['decision']=='suspected'
            assert len(c['prediction']['seizure_events'])==sum(d['decision']=='suspected' for d in decisions)
        if run=='final':
            repairs=json.loads((folder/'boundary_review_log.json').read_text())
            for a in repairs:
                assert (folder/a['trace']).exists()
            parent=json.loads((ROOT/'results/fixed/runs.json').read_text())
            for c in score['cases']:
                base=next(r for r in parent if r['case']==c['case'])
                for k in c['stats']:
                    assert c['stats'][k]==base['stats'].get(k,0)+sum(a['stats'].get(k,0) for a in repairs if a['case']==c['case']),(c['case'],k,'repair accounting')
        checks.append({'run':run,'cases':6,'complete_scans':6,'candidate_count':sum(c['candidate_count'] for c in score['cases']),'invalid_calls':score['tool_statistics']['invalid_calls'],'status':'passed'})
    result={'status':'passed','checks':checks,'note':'Evidence existence, provenance and policy accounting checked. Does not certify medical correctness of model explanations.'}
    save(ROOT/'results/integrity_v2.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
