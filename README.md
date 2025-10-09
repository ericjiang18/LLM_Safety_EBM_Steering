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
python -m pipeline.run_pipeline --config_path configs/llama3.yaml
```


