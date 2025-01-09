import os
import re
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from huggingface_hub import login, HfApi
import torch
import nltk
from scipy.stats import entropy
from tqdm import tqdm
import string
from colorama import Fore, Style

def get_lists_of_candidates(constrast_sets):
  list_and_target = {}
  count_ = 0
  for contrast_set in contrast_sets.values():
    list_and_target[count_] = {'candidates':contrast_set['items'], 'target':contrast_set['target']}
    count_ += 1
  return list_and_target


#TODO: put in utils with generate_oracle_annotations
def get_prompts(*args):
  if type(args[0]) == list:
    if 'llama' in model_id:
        prompt = [
            {'role': "system",
             'content': "You are playing a game, make only one yes/no question at turn to identify the target from the list. "
                        "If there are 1 or 2 candidates remaining make the guess"},
            {'role': "user", 'content': f"List: {', '.join(args[0])}."}]
    else:
        prompt = [
            {'role': "user",
             'content': "You are playing a game, make only one yes/no question at turn to identify the target from the list. "
                        "If there are 1 or 2 candidates remaining make the guess. \n"
                        f"List: {', '.join(args[0])}."}]
  else:
      if 'llama' in model_id:
          prompt = [
              {'role': "system",
               'content': f"You are playing the 20-Questions game, you will be asked one Question about the Target element. "
                          f"Answer only 'yes' or 'no' to the Question depending on your Target Element."},
              {'role': "user", 'content': f"Target element: '{args[1]}'. Question: '{args[0]}'"}
          ]
      else:
          prompt = [
              {'role': "user", 'content': f"You are playing the 20-Questions game, you will be asked one Question about the Target element. "
                                          f"Answer only 'yes' or 'no' to the Question depending on your Target Element."
                                          f"Target element: '{args[1]}'. Question: '{args[0]}'"}
          ]

  return prompt

#TODO: integrate with generate_oracle_annotations (and put to utils)
@torch.no_grad
def llama2_model_call(conversation, oracle=False, repetitive_token_ids=[]):
    if oracle:
        tokenized_conversation = tokenizer(conversation, padding=True, return_tensors="pt").to("cuda")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            sequences = model.generate(**tokenized_conversation, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10,
                                   do_sample=False, num_beams=1, top_p=None, temperature=None)
        return_sequences = tokenizer.batch_decode(sequences[:, tokenized_conversation['input_ids'].shape[1]:], skip_special_tokens=True)
        return_sequences = list(map(lambda answer: "yes" if "yes" in answer.lower() else "no", return_sequences))

    else:
        tokenized_conversation = tokenizer(conversation, return_tensors="pt")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            sequences = model.generate(
                input_ids=tokenized_conversation["input_ids"].to("cuda"),
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=70,
                temperature=1.0,
                do_sample=True,
                top_k=50,
                suppress_tokens=repetitive_token_ids,
            )

        return_sequences = tokenizer.decode(sequences[0][len(tokenized_conversation['input_ids'][0]):], skip_special_tokens=True).split("?")[0]+"?"

    return return_sequences

def repetitive_words_collector(question, stopwords, repetitive_token_ids=[]):
    tokenized_words = tokenizer(question)['input_ids']
    tokenized_text_no_stopwords = list(filter(lambda x: x not in set(stopwords), tokenized_words))
    repetitive_token_ids.extend(tokenized_text_no_stopwords)
    repetitive_token_ids = list(set(repetitive_token_ids))
    return repetitive_token_ids

def compute_question_eig(candidates, single_annotation, answerer_output):
    optimal_change = {
      32: [16], 31: [15, 16], 30: [15], 29: [14, 15], 28: [14], 27: [13, 14],
      26: [13], 25: [12, 13], 24: [12], 23: [11, 12], 22: [11], 21: [10, 11],
      20: [10], 19: [9, 10], 18: [9], 17: [8, 9], 16: [8], 15: [7, 8],
      14: [7], 13: [6, 7], 12: [6], 11: [5, 6], 10: [5], 9: [4, 5], 8: [4],
      7: [3, 4], 6: [3], 5: [2, 3], 4: [2], 3: [1, 2], 2: [1], 1: [0]
    }

    len_objs = len(candidates)

    previous_distr = [1 / len_objs] * len_objs
    previous_etropy = entropy(previous_distr, base=2)
    previous_valid_targets = [1] * len_objs

    previous_distr_per_ans = {}
    previous_entropy_per_ans = {}
    previous_valid_targets_per_ans = {}

    h = {}
    count_ans = {}
    for current_ans in ['yes', 'no']:
      valid_targets = [1 if ans == current_ans else 0 for ans in single_annotation]
      valid_targets = [v if previous_valid_targets[idx_v] else 0 for idx_v, v in
                       enumerate(valid_targets)]
      num_valid_targets = sum(valid_targets)
      if num_valid_targets == 0:
          h[current_ans] = 0
          count_ans[current_ans] = 0
          continue

      prob = 1 / num_valid_targets
      new_distr = []
      for idx, v in enumerate(valid_targets):
          new_distr.append(prob if v == 1 else 0)
      assert (1 - sum(new_distr) < 0.0001)
      new_entropy = entropy(new_distr, base=2)
      h[current_ans] = new_entropy
      count_ans[current_ans] = num_valid_targets
      previous_distr_per_ans[current_ans] = new_distr[:]
      previous_entropy_per_ans[current_ans] = new_entropy
      previous_valid_targets_per_ans[current_ans] = valid_targets[:]

    h_posterior = (count_ans['yes'] / sum(previous_valid_targets)) * h['yes'] + (
          count_ans['no'] / sum(previous_valid_targets)) * h['no']
    eig = previous_etropy - h_posterior

    optimal_eig_question = False
    if sum(previous_valid_targets_per_ans[answerer_output]) in optimal_change[sum(previous_valid_targets)]:
        optimal_eig_question=True

    previous_valid_targets = previous_valid_targets_per_ans[answerer_output]

    return previous_valid_targets, eig, optimal_eig_question


def generate_dialogues_llama(target_list_candidates, path_bootrapped_dialogues):

    if os.path.exists(f"{path_bootrapped_dialogues}/dialogues.txt"):
        with open(f"{path_bootrapped_dialogues}/dialogues.txt", "r") as f:
            dialogues_raw_txt = f.read()
            num_dialogues = len(dialogues_raw_txt.split("******************"))
            target_list_candidates = {key: target_list_candidates[key] for key in target_list_candidates.keys() if int(key) >= (num_dialogues - 1)}

    else:
        if not os.path.exists(f"{path_bootrapped_dialogues}"):
            os.makedirs(f"{path_bootrapped_dialogues}")

    prompt_words = get_prompts([])[0]['content']
    stopwords_nltk = " ".join(nltk.corpus.stopwords.words('english'))
    stopwords = tokenizer(stopwords_nltk+prompt_words+string.punctuation)['input_ids']

    sft_data = []

    for index, value in tqdm(target_list_candidates.items()):

        successful = False
        while not successful:

            dialogue = []

            target = value['target']
            candidates = value['candidates']

            print("******************")
            dialogue.append("******************")
            print(f"target = {target}")
            dialogue.append(f"target = {target}")

            questioner = get_prompts(candidates)

            print('answerer: {}\t'.format(questioner[-1]['content'].strip()))
            dialogue.append('answerer: {}'.format(questioner[-1]['content'].strip()))

            dialogue_stepwise = list(dialogue)

            candidates_validity = {}
            turn = 0
            stop_main_loop = False

            sft_data_dialogue = []

            for interaction in range(20):
                repetitive_token_ids = []
                history = list(questioner)
                questioner_chat = tokenizer.apply_chat_template(questioner,tokenize=False,add_generation_prompt=True)
                tempt_counter = 0
                other_eig_q = []

                if len(candidates) > 1:
                    condition_reached = False
                    while condition_reached == False:
                        question = llama2_model_call(questioner_chat, repetitive_token_ids=repetitive_token_ids)
                        batch_of_prompts = [get_prompts(question, item) for item in candidates]
                        batch_of_prompts = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for
                                               item in batch_of_prompts]

                        oracle_answers = llama2_model_call(batch_of_prompts, oracle=True)
                        answerer_output = oracle_answers[candidates.index(target)]
                        previous_valid_targets, eig, optimal_eig_question = compute_question_eig(candidates, oracle_answers, answerer_output)

                        if optimal_eig_question:
                            candidates_validity[len(candidates_validity)] = previous_valid_targets
                            condition_reached == True
                            best_question_eig = eig
                            print(Fore.GREEN + '-----> Question considered: ', question,
                                  '----- EIG: ', round(eig, 2), Style.RESET_ALL)
                            break
                        else:
                            print(Fore.RED + '-----> Question considered: ', question,
                                  '----- EIG: ', round(eig, 2), Style.RESET_ALL)
                            repetitive_token_ids = repetitive_words_collector(question, stopwords, repetitive_token_ids)
                            other_eig_q.append({'question':question, 'eig':eig})

                        #optional: to print specific candidates' answers related to EIG computation
                        #print(previous_valid_targets, oracle_answers, eig, optimal_eig_question)

                        tempt_counter += 1
                        if tempt_counter >= 20:
                            stop_main_loop = True
                            break
                    if stop_main_loop:
                        break

                else:
                    question = llama2_model_call(questioner_chat)
                    while target not in question:
                        question = llama2_model_call(questioner_chat)

                        tempt_counter += 1
                        if tempt_counter >= 10:
                            stop_main_loop = True
                            break
                    if stop_main_loop:
                        break

                    batch_of_prompts = tokenizer.apply_chat_template(get_prompts(question, candidates[0]), tokenize=False, add_generation_prompt=True)
                    answerer_output = llama2_model_call(batch_of_prompts, oracle=True)[candidates.index(target)]

                candidates_considered = [cand for cand, valid in
                                         zip(candidates, candidates_validity[len(candidates_validity) - 1]) if
                                         valid == 1]

                questioner.append({'role': 'assistant', 'content': re.sub(r"\n\n*", " ", question)})
                questioner.append({'role': 'user', 'content': re.sub("\n", " ", answerer_output)})

                dialogue.append('questioner: {}'.format(questioner[-2]['content'].strip()))
                dialogue.append('answerer: {}'.format(questioner[-1]['content'].strip()))

                #to track conversations
                if turn == 0:
                    dialogue_stepwise.append('questioner: {}'.format(questioner[-2]['content'].strip()))
                else:
                    dialogue_stepwise.append(f"questioner: CANDIDATES: {', '.join(candidates)} \t" \
                                             f"QUESTION: {questioner[-2]['content'].strip()}")
                dialogue_stepwise.append('answerer: {}'.format(questioner[-1]['content'].strip()))
                print(dialogue_stepwise[-2])
                print(dialogue_stepwise[-1])

                dialogue_data_sft_turn = {
                    "dialogue_id": index,
                    "turn": turn,
                    "candidates": candidates,
                    "target":target,
                    "dialogue_history":history,
                    "best_question":question,
                    "best_question_eig":best_question_eig,
                    "other_question_eig":other_eig_q,
                }

                sft_data_dialogue.append(dialogue_data_sft_turn)
                candidates = candidates_considered

                if target.lower() in question.lower() and len(candidates) <= 1:
                    sft_data.extend(sft_data_dialogue)
                    with open(f'{path_bootrapped_dialogues}/dialogues.txt', 'a') as f:
                        for line in dialogue:
                            f.write(f"{line}\n")
                    with open(f'{path_bootrapped_dialogues}/best_dataset_full.json', 'a') as f:
                        for item in sft_data_dialogue:
                            json.dump(item, f)
                            f.write('\n')

                    successful = True
                    break

                if len(candidates) == 1:
                    questioner[-1]['content'] += ". Last remaining element in the list (name it to end the game): " + ", ".join(candidates)
                elif len(candidates) != 1:
                    questioner[-1]['content'] += ". Updated list: " + ", ".join(candidates)
                turn += 1

    return sft_data

def SFT_dataset_format(data):
    chat_dictionary = {}
    for id_dial in data:
        chat_dictionary[id_dial['dialogue_id']] = id_dial['dialogue_history']
        chat_dictionary[id_dial['dialogue_id']].append({'content': f"{id_dial['best_question']}", 'role': 'assistant'})
        chat_dictionary[id_dial['dialogue_id']].append({'content': f"yes", 'role': 'user'})

    with open(f'{path_bootrapped_dialogues}/SFT_dataset_file.jsonl', 'w') as formatted_file:
        for id, dialogue__ in chat_dictionary.items():
            formatted_dialogue = {"messages": dialogue__}
            json.dump(formatted_dialogue, formatted_file)
            formatted_file.write('\n')

#TODO: do it from data object and merge with SFT_dataset_format
def DPO_dataset_format(best_dataset_full_path):
    #the low limit is to prevent using oracle errors, can be changed
    eig_low_limit, eig_high_limit = 0.80, 0.99
    data = []
    with open(best_dataset_full_path) as f:
        for i, line in enumerate(f):
            datum = json.loads(line)
            new_format = []
            for q in datum['other_question_eig']:
                question_eig_dict = {'question': q['question'], 'eig': q['eig']}
                new_format.append(question_eig_dict)
            datum['other_question_eig'] = new_format
            data.append(datum)

    dpo_dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for question_turn in data:
        if question_turn['best_question_eig'] >= eig_high_limit:
            for id_q, question_discarded in enumerate(question_turn['other_question_eig']):
                # TODO:Use filter() instead, see https://huggingface.co/docs/datasets/en/process#select-and-filter:~:text=buffer_size%3D1000)-,Select%20and%20Filter,-There%20are%20two
                if len(tokenizer(question_discarded['question']).input_ids) <= 100 and question_discarded[
                    'eig'] < eig_low_limit:
                    for id, turn in enumerate(question_turn['dialogue_history']):
                        if turn['role'] == 'user':
                            if id != 0 and id != 1:
                                if "Last remaining element in the list (name it to end the game): " in turn['content']:
                                    turn["content"] = turn["content"].replace(
                                        'Last remaining element in the list (name it to end the game): ',
                                        'List: ')
                                elif "Updated List: " in turn['content']:
                                    turn["content"] = turn["content"].replace(
                                        'Updated List: ',
                                        '. Updated List: ')
                                turn["content"] = turn["content"].split(".", 1)[0]
                    dpo_dataset_dict["prompt"].append(
                        tokenizer.apply_chat_template(question_turn['dialogue_history'], tokenize=False,
                                                      add_generation_prompt=False))
                    dpo_dataset_dict["chosen"].append(question_turn['best_question'] + ' ' + tokenizer.eos_token)
                    dpo_dataset_dict["rejected"].append(question_discarded['question'] + ' ' + tokenizer.eos_token)
                else:
                    continue

    return Dataset.from_dict(dpo_dataset_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate dialogues with best EIG.')
    parser.add_argument("--cds", type=str, default="contrast_sets")
    parser.add_argument("--model", type=str, default="llama7b",
                        choices=["llama7b", "llama3_8b", "mistral", "gemma7b"])
    parser.add_argument("--precision", type=str, default="f16",
                        choices=["4bit", "8bit", "f16"])
    args = parser.parse_args()

    model_id = args.model
    models = {
        "llama7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma7b": "google/gemma-1.1-7b-it"
    }

    cache_dir = ""
    login("")

    cache_dir_models, cache_dir_datasets = f"{cache_dir}/hf_llms_checkpoints/", f"{cache_dir}/hf_datasets/"
    huggingface_username = HfApi().whoami()["name"]

    if args.precision == "f16":
        tokenizer = AutoTokenizer.from_pretrained(models[args.model],
                                                  cache_dir=cache_dir_models,
                                                  torch_dtype=torch.float16,
                                                  padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            models[args.model],
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir_models,
            device_map = "auto"
        ).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(models[args.model], cache_dir=cache_dir_models, padding_side="left")
        if args.precision == '8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else: quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            models[args.model],
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
            cache_dir=cache_dir_models,
            device_map="auto"
        ).eval()
    tokenizer.pad_token = tokenizer.eos_token

    path_bootrapped_dialogues = f"./data/bootstrapped/{model_id}/{args.cds}"
    with open(f"./data/game_sets/train/{args.cds}.json") as f:
        contrast_sets = json.load(f)
    target_list_candidates = get_lists_of_candidates(contrast_sets)
    hf_data_full = generate_dialogues_llama(target_list_candidates, path_bootrapped_dialogues)
    SFT_dataset_format(hf_data_full)
    dpo_dataset_format = DPO_dataset_format(f"{path_bootrapped_dialogues}/best_dataset_full.json")
    dpo_dataset_format.push_to_hub(f"{huggingface_username}/LearningToAsk_DPO_{args.cds}")