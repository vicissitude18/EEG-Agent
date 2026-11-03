"""Download upstream code; do not redistribute unlicensed upstream source."""
import subprocess
from pathlib import Path
from paths import ROOT,save
SOURCES={'EEGAgent':('rebootingLine','7ababeaeebe7a89db64eb443d25ddecc280210f0'),
 'EEGAgentBenchmark':('528why','279601a3b689437f2484f50b025ea0fefcbb83ef')}
def main():
    vendor=ROOT/'vendor';vendor.mkdir(exist_ok=True)
    for name,(owner,sha) in SOURCES.items():
        path=vendor/name
        if not path.exists():subprocess.run(['git','clone',f'https://github.com/{owner}/{name}.git',str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',sha],check=True)
    p=vendor/'EEGAgent/tools/__init__.py';s=p.read_text()
    old="import_all_modules(__import__(__name__, fromlist=['']))"
    if "EEGAGENT_BENCH_MODE" not in s:s=s.replace(old,"import os\nif os.environ.get('EEGAGENT_BENCH_MODE') != '1':\n    "+old)
    p.write_text(s)
    p=vendor/'EEGAgent/main.py';s=p.read_text()
    if "self.config.get('enable_rag'" not in s:
        start=s.index('        from RAG.embedder import BGEEmbedder')
        end=s.index('        self.prepare_user_message(user_query)',start)
        s=s[:start]+"        if self.config.get('enable_rag', True):\n"+''.join('    '+line if line.strip() else line for line in s[start:end].splitlines(True))+s[end:]
        p.write_text(s)
    save(ROOT/'sources.json',{k:{'repository':f'https://github.com/{o}/{k}','commit':s} for k,(o,s) in SOURCES.items()})
    print('UPSTREAM_READY')
if __name__=='__main__':main()
