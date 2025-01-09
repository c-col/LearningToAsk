import torch

def oracle_answers(batch_of_prompts, model, tokenizer):
    batch_of_prompts = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for
                        item in batch_of_prompts]
    tokenized_conversation = tokenizer(batch_of_prompts, padding=True, return_tensors="pt").to("cuda")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        sequences = model.generate(**tokenized_conversation, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10,
                                   do_sample=False, num_beams=1, top_p=None, temperature=None)
    return_sequences = tokenizer.batch_decode(sequences[:, tokenized_conversation['input_ids'].shape[1]:],
                                              skip_special_tokens=True)
    return_sequences = list(map(lambda answer: "yes" if "yes" in answer.lower() else "no", return_sequences))
    return return_sequences

def split_batches(samples, batch_size):
    if batch_size > len(samples):
        return [samples]
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batches.append(batch)
    return batches