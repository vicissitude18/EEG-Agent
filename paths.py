import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dependencies():
    candidates=[Path(os.environ['EEG_T5_VENDOR'])] if os.environ.get('EEG_T5_VENDOR') else [ROOT/'vendor',ROOT.parent/'eegagent-bench-mvp/vendor']
    for path in candidates:
        if (path/'EEGAgent/main.py').exists() and (path/'EEGAgentBenchmark/eeg_agent_bench').exists():
            os.environ['EEGAGENT_BENCH_MODE']='1'
            sys.path[:0]=[str(path/'EEGAgent'),str(path/'EEGAgentBenchmark')]
            return path
    raise RuntimeError('Run python bootstrap.py to download pinned upstream code first')
def read_jsonl(p):
    import json
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,x):
    import json
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
