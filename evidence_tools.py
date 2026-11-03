"""Transparent measurement + broad change screening. No reference labels.

Screening is a heuristic attention mechanism, not a seizure diagnosis.
Every retained candidate receives local time/frequency/spatial measurements.
"""
import re,math
import numpy as np
from scipy import signal,stats
from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec

FEATURES=['rms_uv','line_length_uv_s','delta_rel','theta_rel','alpha_rel','beta_rel','gamma_rel','dominant_hz','spectral_entropy','peak_fraction','kurtosis']
def rounded(x):
    if isinstance(x,np.ndarray):return rounded(x.tolist())
    if isinstance(x,(list,tuple)):return [rounded(v) for v in x]
    if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
    if isinstance(x,(float,np.floating)):return round(float(x),4) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
def channel_selection(raw):
    used=[];excluded=[];seen=set()
    for ch in raw.ch_names:
        cleaned=re.sub(r'-\d+$','',ch)
        if not re.fullmatch(r'[A-Za-z]+\d*-[A-Za-z]+\d*',cleaned):excluded.append([ch,'non_bipolar']);continue
        key=tuple(sorted(cleaned.upper().split('-')))
        if key in seen:excluded.append([ch,'duplicate_or_reversed']);continue
        x=raw.get_data(picks=[ch])
        if np.std(x)<1e-12:excluded.append([ch,'flat']);continue
        seen.add(key);used.append(ch)
    return used,excluded
def feature_matrix(x,sf):
    """x in microvolts, channels x samples; Welch features with 1-45Hz denominator."""
    f,p=signal.welch(x,fs=sf,nperseg=min(int(2*sf),x.shape[-1]),axis=-1)
    integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    mask=(f>=1)&(f<=45);total=np.maximum(integrate(p[:,mask],f[mask],axis=-1),1e-12)
    bands=[]
    for lo,hi in [(1,4),(4,8),(8,13),(13,30),(30,45)]:
        m=(f>=lo)&(f<hi);bands.append(integrate(p[:,m],f[m],axis=-1)/total)
    m=(f>=2)&(f<=30);fp=f[m];pp=p[:,m];prob=pp/np.maximum(pp.sum(axis=1,keepdims=True),1e-12)
    dominant=fp[np.argmax(pp,axis=1)]
    entropy=-np.sum(prob*np.log(np.maximum(prob,1e-20)),axis=1)/np.log(len(fp))
    peak=np.array([prob[i,np.abs(fp-dominant[i])<=1].sum() for i in range(len(x))])
    return np.column_stack([np.sqrt(np.mean(x*x,axis=1)),np.mean(np.abs(np.diff(x,axis=1)),axis=1)*sf,*bands,dominant,entropy,peak,stats.kurtosis(x,axis=1)])

class EvidenceEngine:
    def __init__(self,record_id,context,case_dir):
        from eeg_agent_bench.tools.implementations._data_utils import load_raw
        self.case_dir=case_dir;raw=load_raw(record_id,context);self.channels,self.excluded=channel_selection(raw)
        self.sf=float(raw.info['sfreq']);self.duration=raw.n_times/self.sf
        data=raw.get_data(picks=self.channels)*1e6
        sos=signal.butter(4,[1,45],btype='bandpass',fs=self.sf,output='sos')
        self.data=signal.sosfiltfilt(sos,data,axis=1)
        self.starts=np.arange(0,self.duration-4.999,5.)
        self.feat=np.stack([feature_matrix(self.segment(a,min(a+5,self.duration)),self.sf) for a in self.starts])
        self.base=np.maximum(np.median(self.feat[:,:,:2],axis=0),1e-6)
        self.candidates=[];self.reviews={};self.scan_output=None
    def segment(self,a,b):return self.data[:,int(a*self.sf):int(b*self.sf)]
    def nominate(self):
        ratio=self.feat[:,:,:2]/self.base[None,:,:]
        # Broad, fixed signal-change thresholds. No seizure labels or learned weights.
        active=(ratio[:,:,0]>=1.5)&(ratio[:,:,1]>=2.)
        strength=np.percentile(ratio[:,:,1]*np.sqrt(ratio[:,:,0]),90,axis=1)
        bins=np.flatnonzero(np.any(active,axis=1));groups=[]
        for i in bins:
            if groups and i-groups[-1][-1]<=3:groups[-1].append(int(i))
            else:groups.append([int(i)])
        regions=[]
        for group in groups:
            a=max(0,float(self.starts[group[0]])-10);b=min(self.duration,float(self.starts[group[-1]])+15)
            for lo in np.arange(a,b,160):
                hi=min(float(lo)+180,b)
                regions.append({'onset_sec':float(lo),'offset_sec':hi,'screen_strength':float(max(strength[group]))})
        ranked=sorted(regions,key=lambda x:x['screen_strength'],reverse=True)
        self.screen_discarded=max(0,len(ranked)-4)
        self.candidates=[{'candidate_id':f'C{i+1}',**r,'origin':'fixed_broad_change_screen'} for i,r in enumerate(sorted(ranked[:4],key=lambda x:x['onset_sec']))]
        return ratio,strength
    def scan(self):
        if self.scan_output is not None:return self.scan_output
        ratio,strength=self.nominate();rows=[]
        for i in range(0,len(self.starts),2):
            j=min(i+2,len(self.starts));v=self.feat[i:j];rr=ratio[i:j]
            end=min(self.starts[j-1]+5,self.duration);corr=np.corrcoef(self.segment(self.starts[i],end));upper=np.abs(corr[np.triu_indices(len(corr),1)])
            rows.append([self.starts[i],end,np.percentile(rr[:,:,1],95),np.percentile(rr[:,:,0],95),np.percentile(v[:,:,8],25),np.percentile(v[:,:,9],95),int(np.max(np.sum((rr[:,:,0]>=1.5)&(rr[:,:,1]>=2),axis=1))),float(np.nanmean(upper))])
        self.scan_output=rounded({'duration_sec':self.duration,'window_sec':10,'channels_used':self.channels,'excluded_channels':self.excluded,
          'columns':['start','end','LL_ratio_p95','RMS_ratio_p95','entropy_p25','peak_fraction_p95','changed_leads','mean_abs_corr'],
          'rows':rows,'screening_regions':self.candidates,'screening_regions_dropped_by_budget':self.screen_discarded,
          'screen_rule':'5-second bins, any lead RMS >=1.5x and LL >=2x its whole-clip median; merge gap <=10sec, pad10sec; retain top4 regions by LL*sqrt(RMS) contrast. NOT seizure predictions. Agent may add two regions.',
          'measurement_method':'1-45Hz 4th-order zero-phase Butterworth; Welch 2sec; entropy and dominant frequency 2-30Hz. Ratios relative to per-lead clip median.'})
        from paths import save
        save(self.case_dir/'scan.json',self.scan_output);return self.scan_output
    def add_regions(self,regions):
        if len(regions)>2:raise ValueError('At most two additional regions')
        for r in regions:
            a,b=float(r['onset_sec']),float(r['offset_sec'])
            if not 0<=a<b<=self.duration or b-a>180:raise ValueError('Region must be in bounds and <=180 seconds')
            if any(min(b,c['offset_sec'])>max(a,c['onset_sec']) for c in self.candidates):continue
            self.candidates.append({'candidate_id':f'C{len(self.candidates)+1}','onset_sec':a,'offset_sec':b,'origin':'agent_added','reason':r.get('reason','')})
        return {'candidates':self.candidates,'next':'Every candidate must receive measure_candidate then assess_candidate; no candidate can be silently discarded.'}
    def review(self,cid):
        if cid in self.reviews:return self.reviews[cid]
        c=next(v for v in self.candidates if v['candidate_id']==cid);a,b=c['onset_sec'],c['offset_sec']
        lo=max(0,a-60);hi=min(self.duration,b+60)
        mask=(self.starts<b)&(self.starts+5>a)
        outside=((self.starts>=lo)&(self.starts+5<=a))|((self.starts>=b)&(self.starts+5<=hi))
        background=self.feat[outside] if outside.any() else self.feat
        baseline=np.maximum(np.median(background[:,:,:2],axis=0),1e-6)
        during=self.feat[mask];ratio=np.median(during[:,:,:2],axis=0)/baseline
        top=np.argsort(ratio[:,1]*np.sqrt(ratio[:,0]))[-4:][::-1];names=[self.channels[i] for i in top]
        def section(m):
            if not np.any(m):return None
            values=np.median(self.feat[m][:,top,:],axis=0)
            return {name:dict(zip(FEATURES,values[i])) for i,name in enumerate(names)}
        before=(self.starts>=lo)&(self.starts+5<=a);after=(self.starts>=b)&(self.starts+5<=hi)
        pairs=[]
        for i in range(len(top)):
            for j in range(i+1,len(top)):
                def corr(start,end):
                    if end-start<2:return None
                    z=self.segment(start,end)[[top[i],top[j]]]
                    return float(np.corrcoef(z)[0,1])
                shared=bool(set(re.sub(r'-\d+$','',names[i]).split('-'))&set(re.sub(r'-\d+$','',names[j]).split('-')))
                pairs.append({'channels':[names[i],names[j]],'before':corr(lo,a),'during':corr(a,b),'after':corr(b,hi),'shared_electrode':shared})
        series=[]
        for k in np.flatnonzero((self.starts>=lo)&(self.starts<hi)):
            v=self.feat[k,top];rr=v[:,:2]/baseline[top]
            series.append([self.starts[k],min(self.starts[k]+5,self.duration),float(np.median(rr[:,1])),float(np.median(rr[:,0])),float(np.median(v[:,7])),float(np.median(v[:,8])),float(np.median(v[:,9])),int(np.sum((rr[:,0]>=1.5)&(rr[:,1]>=2)))])
        evidence=rounded({'candidate_id':cid,'proposed_interval':[a,b],'review_interval':[lo,hi],
          'channel_selection':'Top4 leads by candidate median LL ratio * sqrt(RMS ratio), using adjacent before/after windows as background; no annotation involved.',
          'channels':names,'evidence_ids':{'temporal':cid+'.temporal','spectral':cid+'.spectral','cross_channel':cid+'.cross_channel'},
          'temporal':{'local_ratios':{self.channels[i]:{'rms_ratio':ratio[i,0],'line_length_ratio':ratio[i,1]} for i in top}},
          'spectral':{'before':section(before),'during':section(mask),'after':section(after)},
          'cross_channel':{'pairwise_correlations':pairs,'caution':'Shared electrodes can create correlation. Correlation alone is not seizure evidence; focal seizures need not be globally synchronous.'},
          'time_series_columns':['start','end','LL_ratio_p50','RMS_ratio_p50','dominant_hz_p50','entropy_p50','peak_fraction_p50','changed_selected_leads'],
          'time_series':series,'caution':'High delta power alone is nonspecific. Compare candidate with before/after, sustained evolution, rhythmicity, and artifact alternatives. A high change score is not a diagnosis.'})
        from paths import save
        save(self.case_dir/f'{cid}_evidence.json',evidence);self.reviews[cid]=evidence;return evidence

class EngineTool(BaseTool):
    def __init__(self,engine,name,schema):self.engine=engine;self.name=name;self.schema=schema
    @property
    def spec(self):return ToolSpec(name=self.name,description='Deterministic EEG measurement / evidence bookkeeping; no reference annotations.',input_schema=self.schema)
    def execute(self,args,context):
        if self.name=='scan_recording':return self.engine.scan()
        if self.name=='propose_additional_regions':
            if not len(args['onsets_sec'])==len(args['offsets_sec'])==len(args['reasons']):raise ValueError('Parallel arrays must have the same length')
            return self.engine.add_regions([dict(onset_sec=a,offset_sec=b,reason=r) for a,b,r in zip(args['onsets_sec'],args['offsets_sec'],args['reasons'])])
        if self.name=='measure_candidate':return self.engine.review(args['candidate_id'])
        raise ValueError('Unknown engine tool')
