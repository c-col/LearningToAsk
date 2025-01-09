# Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain
Code for the EMNLP 2024 paper (Findings).

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

## Citation
If you find it useful, you can cite our paper as: 

```
@inproceedings{mazzaccara2024learningtoask,
    title = "Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain",
    author = "Mazzaccara, Davide  and
      Testoni, Alberto  and
      Bernardi, Raffaella",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-emnlp.291/",
}
```
