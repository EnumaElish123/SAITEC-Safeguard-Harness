# Local Model Files

Put local classifier-head weights, tokenizer files, and model configs here only for local development.

The repository ignores `models/*` by default so large or sensitive model artifacts are not committed. Prefer referencing real paths with environment variables, for example:

```powershell
$env:SAFEGUARD_HEAD_MODEL_PATH="G:\Models\safeguard\classifier_head_v1"
```

If a model is served by an HTTP API, keep the files outside this repository and only configure `configs/providers/*.yaml`.

