import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict
import re
from tqdm import tqdm
import time
import pickle

DEFAULT_SYSTEM_PROMPT = """Your task is to judge the correctness of the knowledge graph triplet. Answer must in English. """

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_models', nargs=4, type=str, required=True,
                        help="Four LoRA model paths for each subgraph")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--data_file', default=None, type=str, required=True,
                        help="JSON file containing triples to evaluate")
    parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--progress_file', default='./progress.pkl', type=str, 
                        help="Path to the pkl file for saving and loading progress")
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--alpha', type=str, default="1.0",
                        help="The scaling factor of NTK method, can be a float or 'auto'. ")
    parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
    parser.add_argument('--load_in_4bit', action='store_true', help="Load the LLM in the 4bit mode")
    parser.add_argument('--use_flash_attention_2', action='store_true',
                        help="Use flash attention to replace the LLaMA attention")
    parser.add_argument('--use_ntk', action='store_true', help="Use dynamic-ntk to extend context window")
    parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="The system prompt of the prompt template.")
    return parser.parse_args()


import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def setup_model_loading(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
    if args.only_cpu:
        args.gpus = ""
        if args.load_in_8bit or args.load_in_4bit:
            raise ValueError("Quantization is unavailable on CPU.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if not args.only_cpu:
        if args.use_flash_attention_2:
            from flash_attn_patch_for_inference import replace_llama_attn_with_flash_attn
            replace_llama_attn_with_flash_attn()
        else:
            from attn_and_long_ctx_patches import apply_attention_patch
            apply_attention_patch(use_memory_efficient_attention=True)
    if args.use_ntk:
        from attn_and_long_ctx_patches import apply_ntk_scaling_patch
        apply_ntk_scaling_patch(args.alpha)

    return load_type, device


def save_progress(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"Progress saved to {file_name}")

def load_progress(file_name):
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        print(f"Progress loaded from {file_name}")
        return data
    except FileNotFoundError:
        print(f"No progress file found. Starting from scratch.")
        return None


def load_models(args, load_type, device):
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)

    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=load_type,
        )
    else:
        quantization_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    models = []
    for lora_model_path in args.lora_models:
        print(f"Loading LoRA model from {lora_model_path}")
        model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=load_type, device_map='auto').half()
        if device == torch.device('cpu'):
            model.float()
        model.eval()
        models.append(model)

    print("All LoRA models loaded successfully.")
    return models, tokenizer


def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})


def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = GenerationConfig(
        temperature=1.2,
        top_k=20,
        top_p=0.95,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=500
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=generation_config
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)

    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    
    return output.split("[/INST]")[-1].strip() if "[/INST]" in output else output


def evaluate_triple(models: List[PeftModel], tokenizer: LlamaTokenizer, triple: Dict, args):
    prompts = [
        f"The triplet is '{{head}}, {{relation}}, {{tail}}'. Is this correct? Explain briefly.",
        f"Evaluate: '{{head}}, {{relation}}, {{tail}}'. Is this correct or incorrect? Provide reasoning.",
        f"Is the following correct: '{{head}}, {{relation}}, {{tail}}'? Justify your answer.",
        f"Consider the triplet '{{head}}, {{relation}}, {{tail}}'. Is it accurate? Explain your conclusion."
        f"Is the triplet '{{head}}, {{relation}}, {{tail}}' correct? You only need to answer 'Incorrect' or 'correct'."
    ]

    prompt_variants = [
        prompt.format(head=triple['head'], relation=triple['relation'], tail=triple['tail']) 
        for prompt in prompts
    ]

    responses = []
    final_responses = []  
    for i, model in enumerate(models):
        response = generate_response(model, tokenizer, prompt_variants[i % len(prompt_variants)], args.device)
        if not response:
            print(f"Model {i+1} failed to generate a response. Storing as an empty string.")
            response = " " 
        decision, labeled_response = extract_decision(response)
        final_responses.append(f"{decision}: {labeled_response.strip()}")
        responses.append(response.strip()) 
        print(f"======= Model {i+1} =======")
        print(f"Input: {prompt_variants[i % len(prompt_variants)]}\n")
        print(f"Output: {response}\n")


    final_decisions = [extract_decision(response)[0] for response in responses]

    print(f"Final decisions extracted: {final_decisions}")

    if len(final_decisions) > 0 and all(decision == final_decisions[0] for decision in final_decisions):
        final_decision = final_decisions[0]
        print(f"Initial consensus reached: {final_decision}.")
    elif len(final_decisions) == 0:
        final_decision = "correct"
        print(f"All responses are empty. Defaulting to 'correct'.")
    else:
        final_decision = None
        print(f"No consensus reached, triggering discussion.")

    return final_responses, final_decision



# def conduct_discussion(models: List[PeftModel], tokenizer: LlamaTokenizer, triple: Dict, args, max_rounds: int = 3):
#     discussion = []
#     previous_responses = ""
#     final_decision = "correct"

#     prompts = [
#         f"Discuss the correctness of the triplet '{{head}}, {{relation}}, {{tail}}'. Provide your reasoning and conclude if it's correct or incorrect.",
#         f"Evaluate whether the triplet '{{head}}, {{relation}}, {{tail}}' is accurate. Explain your answer and state if it's correct or incorrect.",
#         f"Analyze the correctness of the triplet '{{head}}, {{relation}}, {{tail}}'. Give your arguments and decide if it's correct or incorrect.",
#         f"Determine the validity of the triplet '{{head}}, {{relation}}, {{tail}}'. Justify your conclusion and state if it's correct or incorrect."
#     ]

#     for round in range(max_rounds):
#         if round == 0:
#             prompt_variants = [
#                 prompt.format(head=triple['head'], relation=triple['relation'], tail=triple['tail']) 
#                 for prompt in prompts
#             ]
#         else:
#             prompt_template = prompts[round % len(prompts)]
#             prompt = prompt_template.format(
#                 head=triple['head'],
#                 relation=triple['relation'],
#                 tail=triple['tail']
#             )
#             prompt = f"Based on the previous discussion: {previous_responses}, {prompt}"

#         if args.with_prompt:
#             if round == 0:
#                 prompt_variants = [generate_prompt(p, args.system_prompt) for p in prompt_variants]
#             else:
#                 prompt = generate_prompt(prompt, args.system_prompt)

#         responses = []
#         for i, model in enumerate(models):
#             if round == 0:
#                 response = generate_response(model, tokenizer, prompt_variants[i % len(prompt_variants)], args.device)
#             else:
#                 response = generate_response(model, tokenizer, prompt, args.device)
#             responses.append(response)
#             print(f"======= Model {i+1} Round {round + 1} =======")
#             print(f"Input: {prompt if round > 0 else prompt_variants[i % len(prompt_variants)]}\n")
#             print(f"Output: {response}\n")

#         previous_responses = " ".join(responses)
#         discussion.append({"round": round + 1, "responses": responses})

#         final_decisions = [extract_decision(response) for response in responses if response.strip()]

#         if len(final_decisions) > 0 and all(decision == final_decisions[0] for decision in final_decisions):
#             final_decision = final_decisions[0]
#             print(f"Consensus reached: {final_decision}, ending discussion early.")
#             break
#         elif len(final_decisions) == 0:
#             final_decision = "correct"
#             print(f"All responses are empty. Defaulting to 'correct'.")

#     return discussion, final_decision

def conduct_discussion(models: List[PeftModel], tokenizer: LlamaTokenizer, triple: Dict, args, initial_responses: List[str], final_decision: str, max_rounds: int = 3):
    discussion = []
    all_responses = []  

    prompts = [
        f"Discuss the correctness of the triplet '{{head}}, {{relation}}, {{tail}}'. Provide your reasoning and conclude if it's correct or incorrect.",
        f"Evaluate whether the triplet '{{head}}, {{relation}}, {{tail}}' is accurate. Explain your answer and state if it's correct or incorrect.",
        f"Analyze the correctness of the triplet '{{head}}, {{relation}}, {{tail}}'. Give your arguments and decide if it's correct or incorrect.",
        f"Determine the validity of the triplet '{{head}}, {{relation}}, {{tail}}'. Justify your conclusion and state if it's correct or incorrect."
    ]

    for round in range(max_rounds):
        prompt_template = prompts[round % len(prompts)]
        prompt = prompt_template.format(
            head=triple['head'],
            relation=triple['relation'],
            tail=triple['tail']
        )

        if round == 0:
            previous_responses = " ".join([f"Model {i+1} Initial Response: {resp}" for i, resp in enumerate(initial_responses)])
            prompt = f"The initial evaluation indicates the following responses: {previous_responses}. {prompt}"
        else:
            previous_responses = " ".join([f"Model {i+1} Round {r+1}: {resp}" for r, round_responses in enumerate(all_responses) for i, resp in enumerate(round_responses)])
            prompt = f"Based on the previous discussion: {previous_responses}, {prompt}"

        if args.with_prompt:
            prompt = generate_prompt(prompt, args.system_prompt)

        responses = []
        for i, model in enumerate(models):
            response = generate_response(model, tokenizer, prompt, args.device)
            responses.append(response)
            print(f"======= Model {i+1} Round {round + 1} =======")
            print(f"Input: {prompt}\n")
            print(f"Output: {response}\n")

        all_responses.append(responses)
        discussion.append({"round": round + 1, "responses": responses})

        final_decisions = [extract_decision(response) for response in responses if response.strip()]

        if len(final_decisions) > 0 and all(decision == final_decisions[0] for decision in final_decisions):
            final_decision = final_decisions[0]
            print(f"Consensus reached: {final_decision}, ending discussion early.")
            break
        elif len(final_decisions) == 0:
            final_decision = "correct"
            print(f"All responses are empty. Defaulting to 'correct'.")

    return discussion, final_decision




# def extract_decision(response):
#     response = response.lower().strip()
    

#     if not response:
#         return "correct", response

#     if re.search(r'\bcorrect\b', response):
#         return "correct", response
#     elif re.search(r'\byes\b', response):
#         return "correct", response
#     elif re.search(r'\bincorrect\b', response):
#         return "incorrect", response
#     elif re.search(r'\bno\b', response):
#         return "incorrect", response
    
#     # 如果没有匹配到任何信息，返回 "incorrect"
#     return "incorrect", response

def extract_decision(response):
    response = response.lower().strip()
    
    if not response:
        return "correct", response

    if re.search(r'\bcorrect\b', response) or re.search(r'\b是\b', response) or re.search(r'\b对\b', response) or re.search(r'\b正确\b', response):
        return "correct", response
    elif re.search(r'\byes\b', response):
        return "correct", response
    elif re.search(r'\bincorrect\b', response) or re.search(r'\b不是\b', response) or re.search(r'\b不对\b', response) or re.search(r'\b不正确\b', response):
        return "incorrect", response
    elif re.search(r'\bno\b', response):
        return "incorrect", response
    
    return "incorrect", response



def load_and_process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_triples = []
    for item in data:
        input_text = item['input']

        head, relation, tail = [part.strip() for part in input_text.split(',')]
        processed_triples.append({
            "head": head,
            "relation": relation,
            "tail": tail,
            "given_output": item['output']
        })
        print(f"Processed triple: {head}, {relation}, {tail}")  
    return processed_triples


def update_json_file(file_path, result):
    if os.path.exists(file_path):
        with open(file_path, 'r+', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
            existing_data.append(result)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            f.truncate()
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    load_type, device = setup_model_loading(args)
    models, tokenizer = load_models(args, load_type, device)
    args.device = device

    progress_data = load_progress(args.progress_file)

    if progress_data is not None:
        start_index = progress_data['last_index']
    else:
        start_index = 0

    triples = load_and_process_data(args.data_file)
    total_triples = len(triples)

    pbar = tqdm(total=total_triples, desc="Processing triples", unit="triple", initial=start_index)

    start_time = time.time()
    for i, triple in enumerate(triples[start_index:], start=start_index):

        initial_responses, initial_decision = evaluate_triple(models, tokenizer, triple, args)


        if initial_decision is not None:
            final_decision = initial_decision
            discussion = []
        else:

            discussion, final_decision = conduct_discussion(models, tokenizer, triple, args, initial_responses, initial_decision)

            if final_decision is None:
                correct_count = sum(1 for decision in discussion[-1]['responses'] if extract_decision(decision)[0] == "correct")
                incorrect_count = sum(1 for decision in discussion[-1]['responses'] if extract_decision(decision)[0] == "incorrect")
                final_decision = "correct" if correct_count > incorrect_count else "incorrect"

        print(f"Final decision for triple ({triple['head']}, {triple['relation']}, {triple['tail']}): {final_decision}")
        
        result = {
            "triple": f"({triple['head']}, {triple['relation']}, {triple['tail']})",
            "given_output": triple['given_output'],
            "initial_responses": initial_responses,
            "discussion": discussion,
            "final_decision": final_decision
        }

        update_json_file(args.predictions_file, result)

        save_progress({'last_index': i + 1}, args.progress_file)

        pbar.update(1)

        elapsed_time = time.time() - start_time
        triples_per_second = (i + 1) / elapsed_time
        estimated_time_left = (total_triples - (i + 1)) / triples_per_second
        pbar.set_description(f"Processing triples - ETA: {estimated_time_left:.2f}s")

    pbar.close()
    print(f"All results have been saved to {args.predictions_file}")

    if os.path.exists(args.progress_file):
        os.remove(args.progress_file)


if __name__ == "__main__":
    main()
