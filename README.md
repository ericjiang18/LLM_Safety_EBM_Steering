# Energy-Driven Alignment: Reducing False Refusals in Large Language Models


## 🪜 Environment Setup 
```bash
source setup.sh
```
Install the evaluation harness from source

```bash
cd lm-evaluation-harness
pip install -e .
``` 


## 🔭 Experiments 
To run vector extraction, ablation and evaluation, run the script bellow:

```bash
python -m pipeline.run_pipeline --config_path configs/cfg.yaml
```
Before running, please note:​​ Modify the path: "/local3/user/Jailbreaking/dataset/processed/SafeMedEval-21K-all.jsonl"in your cfg.yaml configuration file to the actual path of your SafeMedEval-21K-all.jsonl file.


