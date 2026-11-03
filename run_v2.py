"""EEGAgent inherited loop + real dispatcher, staged native tools and evidence IDs.
No answer-key file is read here or by evidence_tools.py.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from paths import ROOT,dependencies,read_jsonl,save
BENCH=dependencies()/'EEGAgentBenchmark'
from main import EEGAgent
from tools import function_register
from eeg_agent_bench.types import Scenario,AgentAction,ActionType,ToolCallRequest
from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.envs.tool_env import ToolEnvironment
from local_qwen import LocalQwen
from evidence_tools import EvidenceEngine,EngineTool
from support_policy import support_summary
from schema_utils import isolated_schema

SYSTEM='''You are an EEG research tool agent, not a clinical diagnostic system. Use only real tool measurements, never invent tool returns. Reference annotations, original filenames and patient identity are unavailable.
Broad RMS/line-length changes are ATTENTION CANDIDATES, not seizure labels. High delta power, high amplitude, or high correlation alone is insufficient. Reject stable background slow activity and isolated artifacts. Compare before/during/after, sustained rhythmic evolution, clear transition, and spatial support. A focal seizure need not synchronize every channel. Shared electrodes can inflate correlation. If evidence is contradictory, use uncertain instead of inventing support.
Workflow is enforced: scan_recording -> propose_additional_regions -> measure_candidate -> assess_candidate for EVERY candidate -> finish_record. The scan covers the entire clip. Add at most two overlooked regions from the scan, or submit empty parallel arrays. Each local measurement records three evidence IDs. Cite the matching IDs in the assessment. Refine onset/offset using 5-second local rows, within the measured review interval. Only set evolution_supported/context_contrast true when the measured time series supports those claims. All decisions, including rejected/uncertain, will be retained. Give short, concrete reasons referencing measured values. Call exactly one currently offered tool. Do not call unoffered tools or repeat an identical call.'''

def spec(name,properties,required,description):return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}
NUM={'type':'number'};STR={'type':'string'};BOOL={'type':'boolean'}
SCHEMAS={
 'scan_recording':spec('scan_recording',{},[],'Measure the entire record once; return compact 10sec numerical rows and up to four broad change-screen regions. These are not seizure predictions.'),
 'propose_additional_regions':spec('propose_additional_regions',{'onsets_sec':{'type':'array','items':NUM,'maxItems':2},'offsets_sec':{'type':'array','items':NUM,'maxItems':2},'reasons':{'type':'array','items':STR,'maxItems':2}},['onsets_sec','offsets_sec','reasons'],'Add 0-2 nonoverlapping regions missed by the broad screen; each <=180sec. Use empty arrays if no additional region has support. Existing screen candidates are always reviewed.'),
 'measure_candidate':spec('measure_candidate',{'candidate_id':STR},['candidate_id'],'Get real local time-domain, spectral and cross-channel evidence with before/after context for the current candidate.'),
 'assess_candidate':spec('assess_candidate',{'candidate_id':STR,'decision':{'type':'string','enum':['suspected','rejected','uncertain']},'onset_sec':NUM,'offset_sec':NUM,
    'temporal_evidence_id':STR,'spectral_evidence_id':STR,'cross_channel_evidence_id':STR,'temporal_reason':STR,'spectral_reason':STR,'cross_channel_reason':STR,
    'evolution_supported':BOOL,'context_contrast':BOOL,'artifact_suspected':BOOL,'alternative_explanation':STR},
    ['candidate_id','decision','onset_sec','offset_sec','temporal_evidence_id','spectral_evidence_id','cross_channel_evidence_id','temporal_reason','spectral_reason','cross_channel_reason','evolution_supported','context_contrast','artifact_suspected','alternative_explanation'],
    'Record current candidate decision, 3 evidence IDs and short measurement-grounded reasons. suspected requires evolution_supported=true, context_contrast=true and artifact_suspected=false. Use rejected or uncertain when not supported.'),
 'finish_record':spec('finish_record',{},[],'Finish only after all candidates have been measured and assessed; stored decisions form the final result.')}

class EvidenceAgent(EEGAgent):
    def __init__(self,scenario,backend,case_dir,guarded=False):
        self.config={'enable_rag':False};self.backend=backend;self.case_dir=case_dir;self.trace=[];self.decisions=[];self.phase='scan_recording';self.pending=[];self.observations=[];self.final_prediction=None
        self.stats={'model_calls':0,'tool_dispatches':0,'measurements_executed':0,'cache_hits':0,'invalid_calls':0,'submission_rejections':0}
        self.cache={};self.index=0;self.last_error='';self.guarded=guarded
        self.registry=ToolRegistry();self.env=ToolEnvironment(self.registry);self.env.reset(scenario)
        context={'record_paths':{'record':scenario.records[0].data_path},'record_metadata':{'record':scenario.records[0].metadata}}
        self.engine=EvidenceEngine('record',context,case_dir)
        function_register.functions.clear();function_register.metadata.clear()
        for name in ['scan_recording','propose_additional_regions','measure_candidate']:
            schema=SCHEMAS[name]['function']['parameters'];self.registry.register(EngineTool(self.engine,name,schema));function_register.functions[name]=self.wrap(name)
        self.messages=[{'role':'system','content':SYSTEM}]
        self.question=f'Analyze all {self.engine.duration:g} seconds of this EEG clip. Find suspected seizure intervals through local evidence review. Complete the enforced workflow. Do not assume there must be a seizure.'
        self.flush()
    def flush(self):save(self.case_dir/'trace.json',{'system':SYSTEM,'question':getattr(self,'question',''),'events':self.trace,'stats':self.stats,'phase':self.phase})
    def current(self):return self.engine.candidates[self.index] if self.index<len(self.engine.candidates) else None
    def wrap(self,name):
        def execute(**args):
            signature=json.dumps([name,args],sort_keys=True);t=time.time()
            if signature in self.cache:
                self.stats['cache_hits']+=1;result=self.cache[signature]
                self.trace.append({'kind':'cache_hit','name':name,'arguments':args});self.observations.append(result);return result
            try:
                schema=SCHEMAS[name]['function']['parameters']
                if set(args)-set(schema['properties']) or set(schema['required'])-set(args):raise ValueError('Invalid parameters')
                if name!=self.phase:raise ValueError('Tool not offered in this phase')
                if name=='measure_candidate' and args.get('candidate_id')!=self.current()['candidate_id']:raise ValueError('Wrong candidate ID')
                self.stats['tool_dispatches']+=1
                obs=self.env.step(AgentAction(ActionType.TOOL_CALL,tool_calls=[ToolCallRequest('real',name,args)]))[0]
                if obs.status!='ok':raise ValueError(obs.error)
                result=obs.content;self.cache[signature]=result
                self.stats['measurements_executed']+=int(name in ['scan_recording','measure_candidate'])
                self.trace.append({'kind':'tool_result','name':name,'arguments':args,'result':result,'seconds':time.time()-t})
                self.observations.append(result)
                if name=='scan_recording':self.phase='propose_additional_regions'
                elif name=='propose_additional_regions':self.phase='measure_candidate' if self.engine.candidates else 'finish_record'
                else:self.phase='assess_candidate'
                self.last_error='';self.flush();return result
            except Exception as exc:
                result={'error':str(exc)};self.stats['invalid_calls']+=1;self.observations.append(result);self.last_error=str(exc)
                self.trace.append({'kind':'tool_error','name':name,'arguments':args,'error':str(exc)});self.flush();raise
        return execute
    def call_model(self,model=None):
        # Deliberate bounded memory: preserve full audit on disk, expose only the
        # scan during proposal and one local evidence card during assessment.
        context={'duration_sec':self.engine.duration,'phase':self.phase}
        if self.phase=='propose_additional_regions':context['full_scan']=self.engine.scan_output
        if self.phase in ['measure_candidate','assess_candidate']:context['current_candidate']=self.current()
        if self.phase=='assess_candidate':
            context['local_evidence']=self.engine.reviews[self.current()['candidate_id']]
            if self.guarded:
                context['measurement_support']=support_summary(context['local_evidence'])
                context['assessment_instruction']='Interpret the before/during/after evidence critically. Candidate selection itself gives no seizure evidence. Stable high delta with LL/RMS near baseline is rejected. An isolated transient without sustained rhythmic support is rejected or uncertain. Do not claim sustained elevation when measured support bins are absent. A physiological state change may also pass the numerical screen, so passing the guard is not sufficient: compare spectral concentration/entropy/frequency changes and artifact alternatives. Use uncertain when seizure-specific interpretation is not established; do not turn every change into a seizure.'
        if self.phase=='finish_record':context['stored_decisions']=[{'candidate_id':d['candidate_id'],'decision':d['decision']} for d in self.decisions]
        if self.last_error:context['previous_error']=self.last_error
        self.messages=[{'role':'system','content':SYSTEM},{'role':'user','content':self.question+'\n'+json.dumps(context,separators=(',',':'))}]
        # deepcopy preserves aliases: shared STR nodes must be separated before
        # assigning a candidate enum, or the enum leaks into every reason field.
        schema=isolated_schema(SCHEMAS[self.phase])
        if self.guarded and self.phase=='assess_candidate':
            schema['function']['parameters']['properties']['decision']['enum']=context['measurement_support']['allowed_decisions']
        if self.current() and 'candidate_id' in schema['function']['parameters']['properties']:schema['function']['parameters']['properties']['candidate_id']['enum']=[self.current()['candidate_id']]
        if self.phase=='assess_candidate':
            props=schema['function']['parameters']['properties'];ev=context['local_evidence']
            for family in ['temporal','spectral','cross_channel']:
                props[family+'_evidence_id']['enum']=[ev['evidence_ids'][family]]
                props[family+'_reason']={'type':'string','minLength':25,'description':f'Explain actual measured {family} evidence with before/during/after numeric values or change pattern. Do NOT merely repeat the evidence ID.'}
            props['alternative_explanation']={'type':'string','minLength':10,'description':'Discuss background/physiological activity, artifacts or insufficient context; not just a candidate ID.'}
        raw,calls,stats=self.backend.generate(self.messages,[schema]);self.stats['model_calls']+=1
        self.trace.append({'kind':'model_generation','phase':self.phase,'raw_text':raw,**stats});self.pending=calls[:1]
        self.flush()
        if not self.pending:self.last_error='Use one native call to '+self.phase;return raw
        c=self.pending[0];self.messages.append({'role':'assistant','content':'','tool_calls':[{'type':'function','function':c}]})
        print('CALL',self.phase,c['name'],json.dumps(c.get('arguments',{}))[:120],flush=True)
        return '<FUNCTION> '+c['name']+'\n<ARGS> '+json.dumps(c.get('arguments',{}))
    def assess(self,args):
        c=self.current();cid=c['candidate_id'];e=self.engine.reviews[cid]
        if args.get('candidate_id')!=cid:raise ValueError('Wrong candidate ID')
        for family in ['temporal','spectral','cross_channel']:
            if args.get(family+'_evidence_id')!=e['evidence_ids'][family]:raise ValueError('Missing or invented '+family+' evidence ID')
            reason=args.get(family+'_reason')
            if not isinstance(reason,str) or len(reason.strip())<25 or reason.strip()==e['evidence_ids'][family]:raise ValueError('Need a measured '+family+' explanation, not merely an evidence ID')
        decision=args.get('decision')
        if decision not in ['suspected','rejected','uncertain']:raise ValueError('Invalid decision')
        a,b=args.get('onset_sec'),args.get('offset_sec');lo,hi=e['review_interval']
        if not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not lo<=a<b<=hi:raise ValueError('Interval must be inside measured local context')
        if decision=='suspected' and not (args.get('evolution_supported') is True and args.get('context_contrast') is True and args.get('artifact_suspected') is False):raise ValueError('Suspected requires evolution, context contrast and no unresolved artifact suspicion; otherwise use uncertain/rejected')
        record={**args,'evidence_file':cid+'_evidence.json','proposed_interval':e['proposed_interval'],'review_interval':e['review_interval'],'origin':c['origin']}
        if self.guarded:
            support=support_summary(e);record['measurement_support']=support;record['raw_model_assessment']=copy.deepcopy(args)
            duration=b-a;covered=sum(max(0,min(b,y)-max(a,x)) for x,y in support['support_bins'])
            if decision=='suspected' and (not support['meets_minimum_measurement_support'] or duration<10 or covered<max(10,.5*duration)):
                record['decision']='uncertain';record['policy_downgrade_reason']='Claimed interval lacks minimum measured sustained/context support; retained for review, not silently removed.'
        self.decisions.append(record);save(self.case_dir/'decisions.json',self.decisions)
        self.trace.append({'kind':'candidate_assessed','assessment':record});self.index+=1
        self.phase='measure_candidate' if self.current() else 'finish_record';self.last_error=''
    def finish(self):
        if len(self.decisions)!=len(self.engine.candidates):raise ValueError('Unreviewed candidates remain')
        events=[{'onset_sec':d['onset_sec'],'offset_sec':d['offset_sec'],'candidate_id':d['candidate_id'],'evidence_file':d['evidence_file']} for d in self.decisions if d['decision']=='suspected']
        self.final_prediction={'seizure_events':events,'uncertain_candidates':[d['candidate_id'] for d in self.decisions if d['decision']=='uncertain'],'candidates':self.decisions,'boundary_scan_sec':5}
        self.trace.append({'kind':'record_completed','prediction':self.final_prediction});save(self.case_dir/'prediction.json',self.final_prediction)
    def handle_tool_calls(self,response):
        if not self.pending:return True
        c=self.pending[0];name=c['name'];args=c.get('arguments',{})
        if name in ['assess_candidate','finish_record']:
            try:
                if name!=self.phase:raise ValueError('Wrong workflow phase')
                if name=='assess_candidate':self.assess(args)
                else:self.finish();self.flush();return False
            except Exception as exc:
                self.last_error=str(exc);self.stats['submission_rejections']+=1;self.trace.append({'kind':'submission_rejected','error':str(exc),'arguments':args})
            self.flush();return True
        assistant=copy.deepcopy(self.messages[-1]);self.observations=[]
        super().handle_tool_calls(response)
        self.messages[-1]=assistant
        if not self.observations:
            self.last_error='Unknown/unparsed tool call. Use '+self.phase;self.stats['invalid_calls']+=1
        else:self.messages.append({'role':'tool','name':name,'content':json.dumps(self.observations[0])})
        self.flush();return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-path');p.add_argument('--limit',type=int,default=6);p.add_argument('--max-rounds',type=int,default=24);p.add_argument('--output',default='main');p.add_argument('--guarded',action='store_true')
    args=p.parse_args();manifest=json.loads((ROOT/'results/manifest_v2.json').read_text());out=ROOT/'results'/args.output;out.mkdir(parents=True,exist_ok=True)
    scenarios={s['scenario_id']:s for s in read_jsonl(BENCH/'benchmark/scenarios/T5.jsonl')}
    path=args.model_path or json.loads((ROOT.parent/'eeg-agent-demo/results/qwen_download.json').read_text())['snapshot_path']
    backend=LocalQwen(path);rows=[];start=time.time()
    env={'model':'Qwen3-8B','model_revision':Path(path).name,'gpu':backend.torch.cuda.get_device_name(0),'torch':backend.torch.__version__,'model_load_seconds':backend.load_seconds,'max_rounds':args.max_rounds,'inherited_run':EvidenceAgent.run is EEGAgent.run,
      'schema_alias_fix':True,'guarded':args.guarded,'max_new_tokens':768,'max_context_tokens':24576,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'evidence_tools.py',ROOT/'local_qwen.py',ROOT/'support_policy.py',ROOT/'schema_utils.py']},'baseline_comparison':False,'policy':'Full numerical scan + fixed broad change seeds + 0-2 agent additions; mandatory local three-family evidence; bounded stage memory. Optional numeric evidence-sufficiency gate retains weak positives as uncertain.'}
    save(out/'environment.json',env)
    for f in manifest['files'][:args.limit]:
        s=copy.deepcopy(scenarios[f['scenario_id']]);s['scenario_id']='demo';s['gold_private']={};r=s['records'][0];r.update(record_id='record',data_path=f['data_path']);r['metadata'].update(duration_sec=f['duration_sec'],channel_names=f['channel_names'])
        s['input']={'clinical_context':'Research EEG clip','question':'Find suspected intervals using local measurements.'};s['access']['allowed_tools']=['scan_recording','propose_additional_regions','measure_candidate']
        t=time.time();print('CASE_START',f['case'],flush=True)
        row={k:f[k] for k in ['case','subject','scenario_id','duration_sec','source_start_sec']}
        try:
            agent=EvidenceAgent(Scenario.from_dict(s),backend,out/f"case_{f['case']:02d}",guarded=args.guarded)
            result=agent.run(agent.question,max_rounds=args.max_rounds)
            row.update(status='completed' if agent.final_prediction is not None else 'budget_exceeded',prediction=agent.final_prediction,stats=agent.stats,rounds=result['rounds'],candidate_count=len(agent.engine.candidates),screen_dropped=agent.engine.screen_discarded)
        except Exception as exc:row.update(status='failed',prediction=None,error=repr(exc))
        row['elapsed_seconds']=time.time()-t;rows.append(row);save(out/'runs.json',rows);print('CASE_DONE',json.dumps(row)[:400],flush=True)
    env.update(analysis_seconds=time.time()-start,peak_torch_memory_bytes=backend.torch.cuda.max_memory_allocated());save(out/'environment.json',env);print('V2_COMPLETE',flush=True)
if __name__=='__main__':main()
