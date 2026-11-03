"""Fixed six 20-minute demonstration clips from three new subjects.
Selection uses public annotations for stratification, never for inference.
"""
import hashlib,json,shutil,urllib.request
from pathlib import Path
from paths import ROOT,dependencies,read_jsonl,save
REV='7df779ac14573a61be9eb29c6ba2f84bc890c0d0'
PLAN=[('chb02_16+',2400),('chb03_01',1200),('chb05_06',0),('chb02_16+',0),('chb03_01',0),('chb05_06',1200)]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def main():
    import mne
    b=dependencies()/'EEGAgentBenchmark/benchmark'
    runtime={r['scenario_id']:r for r in read_jsonl(b/'runtime/T5.jsonl')}
    base=f'https://huggingface.co/datasets/whhhy123/EEGAgentBenchmark/resolve/{REV}/'
    with urllib.request.urlopen(base+'metadata/files.jsonl',timeout=120) as r:refs={x.get('path',x.get('data_path')):x for x in map(json.loads,r.read().decode().splitlines())}
    data=ROOT/'data';data.mkdir(exist_ok=True);rows=[]
    for case,(name,start) in enumerate(PLAN,1):
        sid='c1_seizure_'+name;rel=runtime[sid]['records'][0]['data_path'];ref=refs[rel];edf=data/(name+'.edf')
        if not edf.exists():
            print('DOWNLOAD',name,flush=True)
            tmp=edf.with_suffix('.partial')
            with urllib.request.urlopen(base+rel,timeout=180) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
            assert sha(tmp)==ref['sha256'];tmp.replace(edf)
        assert sha(edf)==ref['sha256']
        raw=mne.io.read_raw_edf(edf,preload=False,verbose=False)
        raw.crop(start,start+1200,include_tmax=False).load_data();raw.set_meas_date(None)
        target=data/f'clip_{case:02d}_raw.fif';raw.save(target,overwrite=True,verbose=False)
        rows.append({'case':case,'subject':name.split('_')[0],'scenario_id':sid,'source_start_sec':start,'source_end_sec':start+1200,
          'duration_sec':raw.n_times/raw.info['sfreq'],'sampling_rate':raw.info['sfreq'],'channel_names':raw.ch_names,
          'data_path':str(target.resolve()),'source_path':rel,'source_sha256':ref['sha256'],'clip_sha256':sha(target)})
        print('CLIP_READY',case,name,start,raw.n_times/raw.info['sfreq'],flush=True)
    save(ROOT/'results/manifest_v2.json',{'task':'T5-derived','dataset_revision':REV,'selection':'Fixed 20-minute grid blocks; 3 subjects, one annotation-positive and one background block each. Annotation-informed demonstration sampling, NOT blind/random evaluation.',
      'plan_frozen_before_inference':True,'files':rows})
if __name__=='__main__':main()
