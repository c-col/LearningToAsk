from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset, Dataset
import torch
import wandb
import os
import argparse

#TODO: Metric computation for the best number of examples to give (def evaluation() every 1000 examples)
#TODO: accelerate

def apply_chat(dataset, dataset_portion):
    dataset = dataset[dataset_portion].map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with SFT')
    parser.add_argument('--project_name', default='LearningToAsk', help='name of the project for wandb and hf')
    parser.add_argument("--model", type=str, default="llama7b",
                        choices=["llama7b", "llama3_8b", "mistral", "gemma7b"])
    parser.add_argument("--cds", type=str, default="contrast_sets")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_steps', type=float, default=0.1, help='how many steps perform evaluation')
    parser.add_argument('--n_samples', type=int, default=4000, help='number of training samples used')
    parser.add_argument("--batch_size", type=int, default="4")
    args = parser.parse_args()

    cache_dir = ""
    output_dir = f""
    dataset_path = f'data/bootstrapped/{args.model}/{args.cds}/SFT_dataset_file.jsonl'

    models = {
        "llama7b" : "meta-llama/Llama-2-7b-chat-hf",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma7b": "google/gemma-1.1-7b-it"
    }

    tokenizer = AutoTokenizer.from_pretrained(models[args.model], cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        models[args.model],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=f'{cache_dir}/hf_llms_checkpoints/',
        device_map="auto",
    )

    wandb.login()
    project_name, run_name = f'{args.project_name}', f'SFT_{args.n_samples}'
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_LOG_MODEL"] = "end"

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)      #shuffling examples with seed for development
    dataset = dataset.train_test_split(test_size=0.001, shuffle=False)
    dataset['train'] = Dataset.from_dict(dataset['train'][:int(args.n_samples)])
    train_dataset = apply_chat(dataset, 'train')
    eval_dataset = apply_chat(dataset, 'test')

    peft_config = LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "dowpn_proj"],
        modules_to_save=None,
    )

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir=f'{output_dir}/{project_name}',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size//2,
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        eval_steps=args.eval_steps,
        num_train_epochs=1,
        push_to_hub=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        max_seq_length=1024,
        neftune_noise_alpha=5,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="formatted_chat",
    )

    wandb.init(project=project_name, name=run_name)
    trainer.train()
    wandb.finish()