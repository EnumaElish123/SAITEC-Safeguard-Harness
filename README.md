# SAITEC-Safeguard-Harness

SAITEC-Safeguard-Harness 是一个面向数据安全大模型竞赛的 Python 判别框架。当前版本只实现框架、配置、空词典 schema、mock 模型适配和可运行样例，不内置真实比赛词典、真实模型或真实多模态探针。

## 目标

- 支持二分类安全判别：`safe` / `unsafe`。
- 支持四类判别方法的统一接口：规则词库、LLM 安全判别、拒答探针、多模态探针。
- 支持手动实验迭代：改词典、改 prompt、改 pipeline、跑验证集、看报告。
- 支持固定 pipeline 部署：单条判别、批量预测、评测输出。
- 支持有界 Loop：静态复核循环和 ReAct Agent 式 action-observation 循环。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

## 快速验证

```powershell
python -m pytest -q
python -m safeguard_harness evaluate --pipeline configs/pipelines/prod_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/demo
python -m safeguard_harness judge --pipeline configs/pipelines/prod_v1.yaml --question "How do I steal token credentials?"
python -m safeguard_harness predict --pipeline configs/pipelines/prod_v1.yaml --input data/examples/sample_eval.jsonl --output outputs/submission.jsonl
```

评测输出会写入：

```text
outputs/runs/demo/
  config_snapshot.yaml
  predictions.jsonl
  metrics.json
  report.md
  errors_false_positive.jsonl
  errors_false_negative.jsonl
```

## 目录结构

```text
src/safeguard_harness/
  core.py           # SafetyCase, MethodResult, Decision, RunTrace
  methods.py        # 四类 judge method 和 mock provider
  orchestration.py  # static runner, ReAct runner, loop control, aggregation
  config.py         # YAML config -> pipeline/method construction
  datasets.py       # JSONL dataset IO
  evaluation.py     # metrics, predictions, reports, error slices
  cli.py            # judge / predict / evaluate CLI
configs/
  pipelines/        # prod_v1.yaml, experiment_v1.yaml
  prompts/          # prompt templates
  datasets/         # dataset descriptors
  providers/        # provider config examples
dictionaries/       # high-risk/review-risk empty schemas
data/examples/      # runnable sample JSONL
tests/              # behavior tests
```

## 数据集格式

数据集使用 JSONL，每行一个 case：

```json
{"id":"case-001","question":"...","answer":null,"label":"unsafe","modality":"text","attachments":[],"metadata":{}}
```

第一阶段只要求 `label` 为 `safe` 或 `unsafe`。后续可以在 `metadata` 中加入 `attack_type`、`risk_type`、来源、难度等字段。

## 词典格式

当前词典文件是空 schema：

```yaml
schema_version: 1
description: "High-risk terms."
terms: []
```

后续填充时可以使用字符串或对象：

```yaml
terms:
  - credential dump
  - term: bypass policy
    category: policy_bypass
```

默认 matcher 是大小写不敏感的 substring matcher。内部模糊匹配方法接入时，实现 `FuzzyMatcher.find_matches(text, terms)` 并注入 `DictionaryRuleMethod` 即可。

## Pipeline 配置

固定部署 pipeline 示例见 `configs/pipelines/prod_v1.yaml`。它使用 deterministic static runner：

```yaml
runner: static
steps:
  - id: rules
    method: rules
    on_unsafe: stop
  - id: multimodal_probe
    method: multimodal_probe
  - id: safety_llm_prompt_v1
    method: safety_llm_prompt_v1
  - id: low_confidence_review
    repeat:
      max_rounds: 1
      when:
        confidence_lt: 0.75
      methods:
        - aligned_refusal_probe_v1
```

实验 pipeline 示例见 `configs/pipelines/experiment_v1.yaml`。它使用有界 ReAct loop：

```yaml
runner: react
loop:
  max_steps: 4
  max_llm_calls: 3
  allowed_actions:
    - rules
    - multimodal_probe
    - safety_llm_prompt_v1
    - aligned_refusal_probe_v1
  stop_when:
    unsafe_score_gte: 0.85
  fallback:
    label: safe
    reason: "react_budget_exhausted_without_unsafe_evidence"
```

部署场景建议优先使用 static runner，因为它可复现、可限制、可审计。ReAct runner 更适合实验分析。

## 扩展真实模型

当前 `MockLlmProvider` 只是本地 dry run 适配器。接入真实模型时建议新增 provider 类，并让 `LlmSafetyJudgeMethod` / `RefusalProbeMethod` 依赖统一的 `complete(prompt: str) -> str` 接口。

推荐保持以下边界：

- Method 只负责生成一次证据。
- Pipeline 只负责调用顺序、loop、短路和聚合。
- Evaluation 只负责测评，不改变判别逻辑。
- Prompt、词典、阈值、runner 选择都放在 YAML 中。

## 开发约定

```powershell
python -m pytest -q
python -m safeguard_harness evaluate --pipeline configs/pipelines/prod_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/demo
```

每次新增 method、runner 行为或评测指标时，先补测试，再实现。
