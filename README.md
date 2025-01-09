# LearningToAsk
Code for the paper "Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain"

## Setup

If you use conda, create the environment for this project running: 

   ```bash
   conda env create -f environment.yml
   ```

If you use pip, activate your environment and run: 

   ```bash
   pip install -r requirements.txt
   ```

## Bootstrapping 

For creating the datasets for fine-tuning and direct preference optimization, run:

   ```bash
   python scripts/bootstrapping.py 
   ```
