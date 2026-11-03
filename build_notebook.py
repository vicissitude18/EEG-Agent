import json
from paths import ROOT
def main():
    cells=[]
    def add(kind,text):
        cell={'id':f'cell{len(cells):03d}','cell_type':kind,'metadata':{},'source':text.splitlines(True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# 多患者EEG Agent：疑似区间与局部证据复核

**实践B | 3名患者 | 6段×20分钟 | 本地Qwen3-8B | AMD R9700**

这是已保存结果回放，不重新加载大模型。公开标注用于选取每名患者的含发作/背景片段及事后评分，参考起止时间不进入Agent。这是在同一小样本上的功能预演，不能当作独立盲测或完整benchmark。

本页只展示修复schema字段别名问题后的fixed/final结果。早期main、guarded、refined目录保留为调试记录，不能用于评价模型或证明策略提升。不确定候选全部保留，并单独展示复核负担，避免把暂不判断当作误报消失。''')
    add('code','''from pathlib import Path
import json
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()
scores={name:json.loads((ROOT/f'results/{name}/score.json').read_text()) for name in ['fixed','final']}
table='|版本|完成|TP|FP|FN|不确定候选|复核队列TP/FP/FN|\\n|---|---|---|---|---|---|---|\\n'
for name,s in scores.items():
    q=s['inclusive_review_queue'];n=s['candidate_decisions']['uncertain']
    table+=f"|{name}|{s['completed']}/{s['n']}|{s['tp']}|{s['fp']}|{s['fn']}|{n}|{q['tp']}/{q['fp']}/{q['fn']}|\\n"
display(Markdown(table))
display(Markdown('## 数值检查及一次边界复核后的全部6例'))
display(Image(filename=str(ROOT/'results/final/timeline_v2.png')))
''')
    add('markdown','''## 查看每条候选的三类局部证据

可修改案例编号。每条候选都有时域、频谱、跨通道证据ID、测量范围和模型理由；完整数值及前/中/后对照表保存在证据卡。以下优先展示一条疑似候选，也可查看排除和不确定候选。''')
    add('code','''case_id=3  # 1至6；默认展示一次命中的局部证据
candidate_id='C2'  # 可改为该案例内的其他候选；不存在时显示第一条
case=next(c for c in scores['final']['cases'] if c['case']==case_id)
ds=case['prediction']['candidates']
print('患者:',case['subject'],'参考:',case['reference_events'])
print('全部候选:',[(d['candidate_id'],d['decision'],d['onset_sec'],d['offset_sec']) for d in ds])
if ds:
    d=next((d for d in ds if d['candidate_id']==candidate_id),ds[0])
    folder=ROOT/f'results/final/case_{case_id:02d}'
    display(Image(filename=str(folder/(d['candidate_id']+'_local.png'))))
    display(Markdown((folder/(d['candidate_id']+'_证据卡.md')).read_text()))
else:
    print('本例未提出候选，完整扫描仍保存在scan.json。')
''')
    add('markdown','''## 工具调用与下一步

接口按阶段提供合法工具，限制整段重复分析；测量缓存与证据ID检查已有预检。编号对应与数值门槛只解决一部分可靠性问题，不能证明模型给出的医学解释正确。仍需增加患者、伪迹类型、低幅度事件，并用独立固定数据检查误报与漏检的平衡。

完整轨迹见`results/fixed`及`results/final`，复现命令见README。人工核对发现的数值混用、相关性误读与边缘背景缺失见`证据审阅与改进规划.md`；模型文字尚未通过逐项事实认证。本页记录公开仓库所附的云端预演结果。''')
    add('code',"print(json.dumps(scores['final']['tool_statistics'],ensure_ascii=False,indent=2))\n")
    obj={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (ROOT/'demo_v2.ipynb').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('NOTEBOOK_CREATED')
if __name__=='__main__':main()
