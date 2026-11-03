import argparse,base64,io,zipfile
from paths import ROOT
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--compact',action='store_true');parser.add_argument('--clean',action='store_true');args=parser.parse_args()
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.html','.ipynb']]
    files += [p for p in (ROOT/'results').rglob('*') if p.is_file() and p.suffix not in ['.npz','.npy','.fif','.edf']]
    if args.clean:
        allowed={'manifest_v2.json','preflight_fixed.log','run_fixed.log','score_fixed.log','render_fixed.log','refine_final.log','score_final.log','render_final.log','integrity_v2.json','verify_final.log','notebook_check_final.log'}
        files=[p for p in files if 'results' not in p.relative_to(ROOT).parts or p.relative_to(ROOT).parts[1] in ['fixed','final'] or p.name in allowed]
    if args.compact:files=[p for p in files if p.suffix!='.png' or 'final' in p.parts or p.name=='timeline_v2.png']
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    (ROOT/'cloud_v2_export.txt').write_bytes(base64.encodebytes(b.getvalue()))
    print('EXPORT_READY',len(files),len(b.getvalue()))
