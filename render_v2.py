"""Render all completed/failed cases and every candidate, with evidence links."""
import argparse,base64,html,json
from pathlib import Path
from paths import ROOT
def evidence_tables(e):
    groups=[]
    def fmt(x):return '缺失' if x is None else f'{x:.3f}'
    def triple(ch,key):
        return '/'.join(fmt((e['spectral'].get(part) or {}).get(ch,{}).get(key)) for part in ['before','during','after'])
    time_rows=[];freq_rows=[];space_rows=[]
    for ch in e['channels']:
        ratios=e['temporal']['local_ratios'][ch]
        time_rows.append([ch,triple(ch,'rms_uv'),triple(ch,'line_length_uv_s'),fmt(ratios['rms_ratio']),fmt(ratios['line_length_ratio'])])
        freq_rows.append([ch,triple(ch,'dominant_hz'),triple(ch,'spectral_entropy'),triple(ch,'delta_rel'),triple(ch,'peak_fraction')])
    for pair in e['cross_channel']['pairwise_correlations']:
        space_rows.append([' / '.join(pair['channels']),fmt(pair['before']),fmt(pair['during']),fmt(pair['after']),'是' if pair['shared_electrode'] else '否'])
    for label,headers,rows in [
      ('时域：数值顺序为前/中/后',['导联','RMS μV','线长 μV/s','RMS比值','线长比值'],time_rows),
      ('频谱：数值顺序为前/中/后',['导联','主频 Hz','谱熵','delta相对功率','谱峰集中度'],freq_rows),
      ('跨通道：相关性含共享电极标记',['导联对','之前','候选内','之后','共享电极'],space_rows)]:
        md='\n### '+label+'\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n'
        ht='<h3>'+html.escape(label)+'</h3><div style="overflow-x:auto"><table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
        groups.append((md,ht))
    return groups
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--output',default='main');args=p.parse_args();out=ROOT/'results'/args.output
    score=json.loads((out/'score.json').read_text());cases=score['cases'];env=json.loads((out/'environment.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(15,10.5));panels=[];md=['# 多患者20分钟EEG预演：实际运行结果\n',
      '3名患者，6段×20分钟，总2小时。按公开标注选择含发作与背景片段；属于功能预演，不是随机盲测或完整EEGAgentBench成绩。\n',
      f"完成 {score['completed']}/{score['n']}；事件 TP={score['tp']}、FP={score['fp']}、FN={score['fn']}；F1={score['event_based_f1']:.3f}；FP/小时={score['fp_per_hour']:.3f}；阳性片段平均Dice-S={score['dice_s_positive_clips']:.3f}。\n",
      '|案例|患者|参考/片段秒|Agent疑似/片段秒|TP/FP/FN|状态|\n|---|---|---|---|---|---|']
    for ax,c in zip(axes.flat,cases):
        folder=out/f"case_{c['case']:02d}";scan=json.loads((folder/'scan.json').read_text()) if (folder/'scan.json').exists() else {}
        if scan:
            ax.plot([(r[0]+r[1])/120 for r in scan['rows']],[r[2] for r in scan['rows']],color='#255e89',lw=1.1,label='P95 line-length ratio')
            ax.set_yscale('symlog',linthresh=1)
        for a,b in c['reference_events']:ax.axvspan(a/60,b/60,color='#349364',alpha=.35,label='Reference')
        pred=c.get('prediction') or {};decisions=pred.get('candidates',[])
        for d in decisions:
            if d['decision']=='suspected':ax.axvspan(d['onset_sec']/60,d['offset_sec']/60,color='#e69a28',alpha=.42,label='Agent suspected')
            elif d['decision']=='uncertain':ax.axvspan(d['onset_sec']/60,d['offset_sec']/60,color='#8e79c6',alpha=.22,label='Uncertain, not scored positive')
        ax.set_title(f"Case {c['case']:02d} / {c['subject']} | TP {c['event_metrics']['tp']} FP {c['event_metrics']['fp']} FN {c['event_metrics']['fn']}",fontsize=11)
        ax.set_xlim(0,20);ax.set_xlabel('Clip time (minutes)');ax.set_ylabel('LL / clip median');ax.grid(alpha=.2)
        hs,ls=ax.get_legend_handles_labels();uniq=dict(zip(ls,hs))
        if uniq:ax.legend(uniq.values(),uniq.keys(),fontsize=7,loc='upper right')
        predicted=[(e['onset_sec'],e['offset_sec']) for e in pred.get('seizure_events',[])]
        counts='/'.join(str(c['event_metrics'][k]) for k in ['tp','fp','fn'])
        md.append(f"|{c['case']}|{c['subject']}|{c['reference_events']}|{predicted}|{counts}|{c['status']}|")
        cards=[]
        for d in decisions:
            cid=d['candidate_id'];e=json.loads((folder/d['evidence_file']).read_text())
            card=[f"# 案例{c['case']:02d} / 候选{cid}\n",f"决策：**{d['decision']}**；Agent区间：{d['onset_sec']}–{d['offset_sec']}秒；候选原区间：{d['proposed_interval']}；实测复核区间：{d['review_interval']}。\n",
              '以下理由为模型解释，数值证据为真实工具计算；解释是否充分仍可复核。\n',
              '|证据类别|编号|模型理由|\n|---|---|---|']
            for family,label in [('temporal','时域'),('spectral','频谱'),('cross_channel','跨通道')]:card.append(f"|{label}|{d[family+'_evidence_id']}|{d[family+'_reason'].replace('|','/')}|")
            tables=evidence_tables(e)
            card+=['\n替代解释：'+d.get('alternative_explanation','')+'\n']+[v[0] for v in tables]+['## 原始实测证据\n','```json\n'+json.dumps(e,ensure_ascii=False,indent=2)+'\n```']
            (folder/f'{cid}_证据卡.md').write_text('\n'.join(card),encoding='utf-8')
            series=e['time_series'];fig2,aa=plt.subplots(3,1,figsize=(10,7),sharex=True)
            tt=[(v[0]+v[1])/2 for v in series]
            aa[0].plot(tt,[v[2] for v in series],label='Median LL / adjacent background');aa[0].plot(tt,[v[3] for v in series],label='Median RMS / adjacent background');aa[0].set_ylabel('Ratio');aa[0].legend(fontsize=7)
            aa[1].plot(tt,[v[4] for v in series],color='#256397',label='Dominant frequency');aa[1].set_ylabel('Hz');aa[1].legend(fontsize=7)
            aa[2].plot(tt,[v[5] for v in series],label='Spectral entropy');aa[2].plot(tt,[v[6] for v in series],label='Peak fraction');aa[2].set_ylabel('Fraction');aa[2].legend(fontsize=7)
            for a in aa:
                a.axvspan(e['proposed_interval'][0],e['proposed_interval'][1],alpha=.12,color='#708697');a.axvspan(d['onset_sec'],d['offset_sec'],alpha=.15,color='#e69a28');a.grid(alpha=.2)
            aa[-1].set_xlabel('Clip time (seconds)');fig2.suptitle(f"Case {c['case']:02d} {cid}: {d['decision']} | local evidence");fig2.tight_layout();fig2.savefig(folder/f'{cid}_local.png',dpi=130);plt.close(fig2)
            pic=(folder/f'{cid}_local.png').relative_to(ROOT).as_posix()
            cardhtml='<details><summary>'+html.escape(f"{cid} | {d['decision']} | {d['onset_sec']}–{d['offset_sec']}秒")+'</summary>'
            cardhtml+=f'<img src="{pic}">'
            for family,label in [('temporal','时域'),('spectral','频谱'),('cross_channel','跨通道')]:cardhtml+=f'<p><b>{label} / {html.escape(d[family+"_evidence_id"])}</b>：{html.escape(d[family+"_reason"])}</p>'
            cardhtml+='<p>替代解释：'+html.escape(d.get('alternative_explanation',''))+'</p>'+''.join(v[1] for v in tables)+'<pre>'+html.escape(json.dumps(e,ensure_ascii=False,indent=2))+'</pre></details>';cards.append(cardhtml)
        panels.append(f'<section><h2>案例{c["case"]:02d} / {c["subject"]}</h2><p>状态：{c["status"]}；参考区间：{c["reference_events"]}；疑似区间：{predicted}；TP/FP/FN：{counts}</p>'+''.join(cards)+'</section>')
    fig.tight_layout();fig.savefig(out/'timeline_v2.png',dpi=150);plt.close(fig)
    overview={k:v for k,v in score.items() if k!='cases'}
    md+=['\n## 工具调用统计\n','```json\n'+json.dumps(score['tool_statistics'],ensure_ascii=False,indent=2)+'\n```',
      '\n每条候选（含排除、不确定）均保留3类局部测量、对应证据编号和模型理由，见各case目录中的证据卡。\n',
      f"冷加载{env['model_load_seconds']:.1f}秒；分析{env['analysis_seconds']:.1f}秒；峰值torch分配{env['peak_torch_memory_bytes']/1024**3:.2f}GiB。\n",
      '本页主结果只使用修复schema字段共享错误后的运行。早期调试运行不能用于评价模型或证明策略提升。本轮不做正式baseline比较。候选筛选有固定阈值与预算，可能漏掉幅度变化不明显的发作。事件评分容差不代表边界精确，需结合Dice与区间图。\n']
    (out/'运行结果.md').write_text('\n'.join(md),encoding='utf-8')
    pic=(out/'timeline_v2.png').relative_to(ROOT).as_posix()
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多患者EEG Agent：候选证据复核</title><style>body{margin:0;background:#f3f6fa;color:#18354c;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;padding:28px 20px;margin:auto}img{max-width:100%;height:auto}section,details{background:white;padding:18px;margin:15px 0;border:1px solid #d4dfe7;border-radius:8px}summary{cursor:pointer;font-weight:700}pre{font:12px/1.6 monospace;white-space:pre-wrap;word-break:break-word;max-height:450px;overflow:auto}h1{line-height:1.4}</style><main><h1>多患者20分钟EEG：候选与局部证据</h1><p>3名患者，6段×20分钟。公开标注用于选取展示片段及事后核对，起止答案不进入Agent推理。不是盲测、完整benchmark或临床诊断。</p><p>蓝线为线长变化比；绿带为参考区间；橙带为Agent疑似；紫带为不确定（未作为阳性计分）。宽松筛选候选不直接等于发作；每条候选必须完成局部时域、频谱、跨通道证据复核。</p>'''
    page=page.replace('</style>','table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #d4dfe7;padding:7px;text-align:left}th{background:#edf3f7}</style>')
    page+=f'<img src="{pic}"><h2>全部案例汇总</h2><pre>'+html.escape(json.dumps(overview,ensure_ascii=False,indent=2))+'</pre>'+''.join(panels)+'</main></html>'
    (ROOT/'demo_v2.html').write_text(page,encoding='utf-8');print('V2_RENDER_COMPLETE')
if __name__=='__main__':main()
