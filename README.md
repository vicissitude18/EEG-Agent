# 基于脑电的疑似癫痫发作区间筛查智能体设计和云端实践

这个程序从多导联脑电提取时域、频谱和跨导联证据，Qwen3-8B 据此筛查疑似癫痫发作区间。云端预演在 AMD AUP 的 Radeon AI PRO R9700（32 GB）上完成。

最终评测覆盖 3 名患者的 6 段记录，每段 20 分钟。严格疑似结果为命中 2 次、误报 2 次、漏检 1 次，事件精确率 50.0%、召回率 66.7%、F1 57.1%。这组成绩来自预演版本。

- [结果、六例时间线和候选证据卡](结果与当前问题.md)
- [可回放的 Notebook](demo_v2.ipynb)
- [证据错误审阅及后续方案](证据审阅与改进规划.md)

## 固定数据

3名新患者chb02、chb03、chb05，每人2段、每段1200秒，共2小时。使用固定20分钟网格块，按公开标注选取每人一段含发作、一段背景；推理前固定清单。这是定向功能预演，不是随机盲测、临床验证或完整EEGAgentBench成绩。模型看不到患者、文件名、来源时间偏移及参考起止时间。

| 案例 | 原始记录 | 原始时间块/秒 |
|---|---|---|
| 01 | chb02_16+ | 2400–3600 |
| 02 | chb03_01 | 1200–2400 |
| 03 | chb05_06 | 0–1200 |
| 04 | chb02_16+ | 0–1200 |
| 05 | chb03_01 | 0–1200 |
| 06 | chb05_06 | 1200–2400 |

信号来自固定HuggingFace release（revision见prepare_v2.py），下载后SHA-256校验。选取流程知道公开标注，推理代码及测量工具不读答案。score_v2.py在预测落盘后独立读取公开答案。

## 本轮改进

1. 物理双极导联去重：保留一条T8-P8自动后缀重复通道，合并P7-T7/T7-P7等反向重复，记录排除原因。
2. 完整数值扫描：5秒底层特征，10秒压缩表。RMS和线长按各导联自身全片段中位数归一化，避免只把慢波功率高当成发作。
3. 宽松候选筛选：固定RMS/线长变化阈值，保留最多4条较强变化区域；Agent还可根据完整扫描表增加2条。阈值是注意力筛选规则，不是癫痫分类器，可能遗漏幅度变化小的发作；被预算丢弃区域数量记录在日志。
4. 每条候选必做局部复核：候选及前后各60秒，选择变化较明显的4导联；记录RMS、线长、频带功率、主频、谱熵、谱峰集中度及导联相关性。共享电极导致的相关性单独标记。
5. 证据对应：时域、频谱、跨通道各有唯一编号，Agent必须引用正确编号并解释保留/排除/不确定。疑似结论需声明持续演化、与背景对比及伪迹考虑；这些声明仍是模型判断，编号有效不保证解释正确。
6. 减少无效调用：当前阶段只提供下一步合法工具；重复测量缓存；参数与证据编号检查；推理只携带当前阶段需要的记忆，完整轨迹另存，避免反复输入全部历史。

第二轮`guarded`增加数值支持检查：局部连续至少15秒、线长比值≥2、RMS比值≥1.5、至少2条所选导联变化，并满足谱峰集中度与谱熵的最低条件；具体见support_policy.py。这是固定工程门槛，没有经过临床验证，可能阻拦真实低幅度或局灶事件。即使模型声称疑似，若所给区间没有足够测量支持，也会保留原始模型判断并降为uncertain。模型解释、数值检查结果、降级原因均保留。

最终主结果仅使用`results/fixed`及其边界复核后的`results/final`，严格疑似与“疑似+不确定”复核队列同时计分。

早期`main`、`guarded`、`refined`运行发现schema字段对象共享错误：给candidate_id增加enum时，限制传播到了理由等字段。它们保留为调试记录，不能用于评价模型能力或证明策略改善。修复使用JSON方式隔离schema节点，并加入独立回归检查、证据ID专属枚举及说明字段长度检查。最终交付包不包含这些无效调试运行，原始记录仍保留在工作目录及AUP。

`refine_v2.py`对全部满足数值最低支持、但仍不确定的候选做一次局部与边界复核，只读取已保存测量，不读取答案或分数，也不重新计算信号特征。结果另存`results/final`，原始判断和复核轨迹均保留。它不保证追回漏检，实际结果见score.json。

`EvidenceAgent.run`直接继承EEGAgent，测量通过原始handle_tool_calls分发器执行；新增工具实现原创，官方事件评分来自EEGAgentBench。当前不是原始benchmark工具设置下的排行榜复现。

## AUP运行

以下命令适用于已配置 ROCm 和 Qwen3-8B 快照的原始 AUP 工作目录。公开仓库随附 `fixed` 和 `final` 结果，可用 `python3 -X utf8 verify_v2.py` 核对；重新推理时请使用不同的输出目录名。

```bash
cd /path/to/EEG-Agent
python3 prepare_v2.py
python3 check_v2.py
python3 run_v2.py --guarded --output fixed-replay
python3 score_v2.py --output fixed-replay
python3 render_v2.py --output fixed-replay
python3 refine_v2.py --source fixed-replay --output final-replay
python3 score_v2.py --output final-replay
python3 render_v2.py --output final-replay
```

新环境先安装requirements（保留平台ROCm torch），运行bootstrap.py下载固定上游代码，再用`--model-path`传入model_source.json规定的Qwen3-8B快照路径。默认模型路径读取相邻目录 `../eeg-agent-demo/results/qwen_download.json` 的 `snapshot_path`。边界复核也使用该路径；如果要在新环境重跑边界复核，请创建此文件并填入本地模型快照的绝对路径。每轮最多24次生成，每次最多768输出token，输入上限24576。

再次测试使用`python3 run_v2.py --guarded --output 新目录名`，然后对score_v2.py和render_v2.py传入相同`--output`，保留既有结果。final目录若已存在，边界复核脚本会拒绝覆盖。

结果包括：全部6例区间图、运行结果Markdown、每条候选的局部PNG、证据卡Markdown、原始证据JSON、模型调用轨迹、最终决策及算力统计。`uncertain`不作为阳性计分，原样保留以便人工复查。score.json还标记参考事件是否被候选阶段覆盖，帮助区分筛选漏检和决策漏检。

结束后在AUP主页Stop My Server。只回放已保存结果可用CPU环境。

HTML使用相对图片资源，需要保留完整目录后打开；Notebook已保存输出。紧凑交付包保留最终全部局部PNG及主结果总览；fixed阶段原始测量、模型输出和证据卡完整保留，可在AUP重新绘图。
