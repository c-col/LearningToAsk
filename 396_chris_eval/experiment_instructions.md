# 20 Questions Experiment Instructions

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/c-col/LearningToAsk.git
cd 396_chris_eval
```

2. Create and activate conda environment:
```bash
conda create -n 20_qns python=3.11
conda activate 20_qns
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiment Matrix

### Model Variants
- DeepSeek R1 Distill Qwen 1.5B (VLC)
- DeepSeek R1 Distill Qwen 7B (MKA)
- DeepSeek R1 Distill Qwen 14B (ZNU)
- DeepSeek R1 Distill Qwen 32B (LDR)

### Datasets
- `contrast_sets_8_celebs.json`: 8-entity celebrity sets (N=90)
- `contrast_sets_8_things.json`: 8-entity object sets (N=90)
- `contrast_sets_8.json`: General 8-entity sets (N=90)
- `contrast_sets_16.json`: 16-entity sets (N=90)
- `contrast_sets_bigbench.json`: BigBench-derived sets (N=29)

### Experiment Tracking

| Model | Dataset | Priority | Person Running |
|-------|---------|----------|----------------|
| DeepSeek R1 Distill Qwen 1.5B (VLC) | contrast_sets_8_celebs.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 1.5B (VLC) | contrast_sets_8_things.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 1.5B (VLC) | contrast_sets_8.json | high | Simon |
| DeepSeek R1 Distill Qwen 1.5B (VLC) | contrast_sets_16.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 1.5B (VLC) | contrast_sets_bigbench.json | high | Shubham |
| DeepSeek R1 Distill Qwen 7B (MKA) | contrast_sets_8_celebs.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 7B (MKA) | contrast_sets_8_things.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 7B (MKA) | contrast_sets_8.json | high | Simon |
| DeepSeek R1 Distill Qwen 7B (MKA) | contrast_sets_16.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 7B (MKA) | contrast_sets_bigbench.json | high | Shubham |
| DeepSeek R1 Distill Qwen 14B (ZNU) | contrast_sets_8_celebs.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 14B (ZNU) | contrast_sets_8_things.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 14B (ZNU) | contrast_sets_8.json | high | Simon |
| DeepSeek R1 Distill Qwen 14B (ZNU) | contrast_sets_16.json | high | Shubham |
| DeepSeek R1 Distill Qwen 14B (ZNU) | contrast_sets_bigbench.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 32B (LDR) | contrast_sets_8_celebs.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 32B (LDR) | contrast_sets_8_things.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 32B (LDR) | contrast_sets_8.json | high | Simon |
| DeepSeek R1 Distill Qwen 32B (LDR) | contrast_sets_16.json | TBD | TBD |
| DeepSeek R1 Distill Qwen 32B (LDR) | contrast_sets_bigbench.json | high | Shubham |

## Running Experiments

### Base Command Format
```bash
python play_20qns_api.py -g <model-name> -gpe -gt r1 --dataset-path "../data/game_sets/test/<dataset-file>"
```

### Experiment Commands

#### 1.5B Model (VLC)
```bash
# Celebrities dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-1-5-vlc -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_celebs.json"

# Objects dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-1-5-vlc -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_things.json"

# General 8-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-1-5-vlc -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8.json"

# 16-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-1-5-vlc -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_16.json"

# BigBench dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-1-5-vlc -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_bigbench.json"
```

#### 7B Model (MKA)
```bash
# Celebrities dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-7b-mka -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_celebs.json"

# Objects dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-7b-mka -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_things.json"

# General 8-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-7b-mka -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8.json"

# 16-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-7b-mka -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_16.json"

# BigBench dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-7b-mka -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_bigbench.json"
```

#### 14B Model (ZNU)
```bash
# Celebrities dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-14b-znu -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_celebs.json"

# Objects dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-14b-znu -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_things.json"

# General 8-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-14b-znu -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8.json"

# 16-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-14b-znu -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_16.json"

# BigBench dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-14b-znu -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_bigbench.json"
```

#### 32B Model (LDR)
```bash
# Celebrities dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-32b-ldr -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_celebs.json"

# Objects dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-32b-ldr -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8_things.json"

# General 8-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-32b-ldr -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_8.json"

# 16-entity dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-32b-ldr -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_16.json"

# BigBench dataset
python play_20qns_api.py -g deepseek-r1-distill-qwen-32b-ldr -gpe -gt r1 --dataset-path "../data/game_sets/test/contrast_sets_bigbench.json"
```

## Output Structure
Results will be saved in:
```
data/game_sets/test/outputs/results_<dataset>_<model>/
├── checkpoints/      # Individual game results
│   ├── game_0.json
│   ├── game_1.json
│   └── ...
└── game_results.json # Combined results and config
```

