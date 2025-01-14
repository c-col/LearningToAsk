from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from huggingface_hub import login, HfApi
from scripts.modified_dpo_data_collator import DPODataCollatorWithPadding
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset
import torch
import wandb
import os
import argparse

def padding_token(tokenizer, model):
    if '<pad>' not in tokenizer.get_vocab():
        added_tokens = tokenizer.add_special_tokens({"pad_token" : "<pad>"})
    else:
        added_tokens = 0

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return(tokenizer, model)

#TODO: check
def get_dataset(hf_dataset):
    dpo_train = load_dataset(hf_dataset, cache_dir=f'{cache_dir}/hf_datasets/')
    dpo_train = dpo_train.filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length
                  and len(x["prompt"]) + len(x["rejected"]) <= max_length)
    return dpo_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with DPO')
    parser.add_argument('--project_name', default='LearningToAsk', help='name of the project for wandb and hf')
    parser.add_argument('--run_name', default='DPO', help='name of the project for wandb and hf')
    parser.add_argument("--model", type=str, default="llama7b",
                        choices=["llama7b", "llama3_8b", "mistral", "gemma7b"])
    parser.add_argument("--SFT_adapter", type=str, help='name of the SFT model for DPO')
    parser.add_argument("--hf_dataset", default='LearningToAsk_DPO_contrast_sets', type=str)
    parser.add_argument("--batch_size", type=int, default="4")
    parser.add_argument('--dataset_portion', type=float, default=100)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--beta', type=float, default=0.1, help='dpo beta value')
    parser.add_argument('--loss_type', choices=['sigmoid', 'hinge', 'ipo', 'kto_pair'], default='sigmoid', help='dpo loss type')
    parser.add_argument('--eval_steps', type=float, default=0.1, help='how many steps perform evaluation')
    parser.add_argument("--precision", type=str, default="f16",
                        choices=["4bit", "8bit", "f16"])
    args = parser.parse_args()

    cache_dir = ""
    output_dir = ""
    login("")
    huggingface_username = HfApi().whoami()["name"]

    model_id, hf_dataset, batch_size = args.model, f'{huggingface_username}/{args.hf_dataset}', args.batch_size

    models = {
        "llama7b" : "meta-llama/Llama-2-7b-chat-hf",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    }

    if args.precision == "f16":
        tokenizer = AutoTokenizer.from_pretrained(models[args.model],
                                                  cache_dir=cache_dir)

        model = AutoModelForCausalLM.from_pretrained(
            models[args.model],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            cache_dir=f'{cache_dir}/hf_llms_checkpoints/',
            device_map = "auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(models[args.model], cache_dir=cache_dir, padding_side="left")
        if args.precision == '8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else: quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            models[args.model],
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
            cache_dir=f'{cache_dir}/hf_llms_checkpoints/',
            device_map="auto"
        )

    if model_id == 'llama3_8b':
        tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
    tokenizer, model = padding_token(tokenizer, model)

    if args.SFT_adapter:
        peft_model_id = f"{huggingface_username}/{args.SFT_adapter}"
        model = PeftModel.from_pretrained(model, peft_model_id)
        model.merge_and_unload()

    wandb.login()
    project_name, run_name = args.project_name, f'{args.run_name}'
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_LOG_MODEL"] = "end"

    max_length = 1024

    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        max_prompt_length=max_length
    )
    dataset = get_dataset(hf_dataset)          
    dataset = dataset.shuffle(seed=42)
    dataset = dataset['train'].train_test_split(test_size=0.01, shuffle=False)

    dataset['test'] = Dataset.from_dict(dataset['test'][:round(len(dataset['test']) * args.dataset_portion/100)])
    dataset['train'] = Dataset.from_dict(dataset['train'][:round(len(dataset['train']) * args.dataset_portion/100)])

    peft_config = LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=None,
    )

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{project_name}/",
        report_to="wandb",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=(batch_size+1)//2,
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_accumulation_steps=8//batch_size,
        gradient_checkpointing=True,
        num_train_epochs=1,
        push_to_hub=True,
        remove_unused_columns=False,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_length,
        peft_config=peft_config,
        data_collator=data_collator,
        loss_type=args.loss_type,
        generate_during_eval=True
    )

    wandb.init(project=project_name, name=run_name)
    dpo_trainer.train()
    dpo_trainer.save_model()
    wandb.finish()