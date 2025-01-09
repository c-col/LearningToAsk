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

To create the datasets for fine-tuning and Direct Preference Optimization (DPO), insert the huggingface_login credential and the path to save HF models and datasets in lines 385-386 of ```bootstrapping.py```. Then run it:

   ```bash
   python bootstrapping.py 
   ```
This will populate the ```data/bootstrapped``` folder and create a HuggingFace dataset that will be used for DPO ([DPO dataset](https://huggingface.co/datasets/mazzaqq/LearningToAsk_DPO_contrast_sets) used in the paper).
