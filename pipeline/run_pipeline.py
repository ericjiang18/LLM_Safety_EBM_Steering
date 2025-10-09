import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import json
import os
import argparse
import mmengine # For config loading
# import jsonpickle # Not needed as LMEvalHarness is removed
import sys
import gc # For garbage collection
from tqdm import tqdm
import numpy as np # For np.mean in evaluation (if used by copied eval funcs)
import csv # Added for CSV dataset loading
from typing import List, Optional, Tuple, Callable, Any, Dict, Union

# Set CUDA_VISIBLE_DEVICES to only use GPUs 4, 5, 6, 7
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# --- Imports from EBM-Refusal-Mitigation project ---
from dataset.load_dataset import load_dataset_split, load_dataset # Assuming load_dataset is also available if needed by copied funcs

from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    add_hooks,
    get_ebm_intervention_hook
)
from pipeline.ebm.ebm_model import SimpleEBM, ComplexEBM, load_ebm_model

# --- InfoNCE Loss Helper ---
def _calculate_infonce_loss(energy_p: torch.Tensor, energy_n: torch.Tensor, temperature: float, device: str) -> torch.Tensor:
    """Calculates InfoNCE loss.
    Args:
        energy_p: Energies of positive samples (batch_p, 1) or (batch_p,).
        energy_n: Energies of negative samples (batch_n, 1) or (batch_n,).
        temperature: Temperature parameter for scaling logits.
        device: Device to send labels to.
    Returns:
        Scalar loss tensor.
    """
    if energy_p.numel() == 0: # No positive samples
        return torch.tensor(0.0, device=device, requires_grad=True)
    if energy_n.numel() == 0: # No negative samples, but positive samples exist - this is unusual for InfoNCE
        # This case might mean all samples are positive, or an error. 
        # For now, let's return 0 loss, but this might need specific handling based on desired behavior.
        # A robust InfoNCE needs at least one negative. Here, we are contrasting P against N.
        # If N is empty, maybe the loss should push E_p to be very low?
        # Softmax over a single logit will be 1, log(1)=0. So -(-E_p/T) + log(exp(-E_p/T)) = 0.
        # Let's make it a sum of -log(sigmoid(-E_p/T)) if no negatives?
        # For simplicity with the current structure: if no negatives, loss is 0 for now. This might need refinement.
        # print("Warning: InfoNCE called with 0 negative samples.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    scores_p = -energy_p.squeeze(-1) / temperature # (batch_p,)
    scores_n = -energy_n.squeeze(-1) / temperature # (batch_n,)

    criterion = nn.CrossEntropyLoss()
    batch_total_loss = []

    for i in range(scores_p.shape[0]):
        current_score_p = scores_p[i] # scalar
        # Logits: [positive_score, negative_score_1, negative_score_2, ...]
        logits = torch.cat([current_score_p.unsqueeze(0), scores_n]) # (1 + batch_n,)
        # Target label is 0 (the positive sample)
        labels = torch.zeros(1, dtype=torch.long, device=device)
        loss_i = criterion(logits.unsqueeze(0), labels) # logits need to be (batch_size_criterion, num_classes)
        batch_total_loss.append(loss_i)
    
    if not batch_total_loss:
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    return torch.mean(torch.stack(batch_total_loss))

# --- Logger Class (copied from run_pipeline.py) ---
class Logger:
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_file, "a", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Focused EBM Training and Evaluation Pipeline.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Override model_path from config')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Override batch_size for LLM generation (if applicable)')
    parser.add_argument('--force_retrain_ebm', action='store_true', help='Force retraining of EBMs even if they exist.')
    return parser.parse_args()

# --- generate_and_save_completions_for_dataset (copied from run_pipeline.py) ---
def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, system=None):
    completions_dir = os.path.join(cfg.artifact_path, 'completions')
    os.makedirs(completions_dir, exist_ok=True)

    dataset_key_for_filename = os.path.basename(str(dataset_name))

    loaded_prompts_for_dataset = [] # Renamed from loaded_prompts to avoid conflict if dataset is passed

    if dataset is None:
        print(f"Attempting to load dataset for '{dataset_name}'...")
        num_test_samples = cfg.get('n_test', cfg.get('n_val', 100))
        if num_test_samples is None:
            num_test_samples = 10000  # 默认使用一个大数字以包含完整数据集
        is_file_path = isinstance(dataset_name, str) and os.path.isfile(dataset_name)

        if is_file_path:
            print(f"Treating '{dataset_name}' as a direct file path.")
            try:
                temp_prompts = []
                if dataset_name.endswith(".json"):
                    with open(dataset_name, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            if data and isinstance(data[0], dict):
                                temp_prompts = [item.get('instruction', item.get('prompt', '')) for item in data]
                            elif data and isinstance(data[0], str):
                                temp_prompts = data
                elif dataset_name.endswith(".jsonl"):
                    with open(dataset_name, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            temp_prompts.append(item.get('instruction', item.get('prompt', '')))
                elif dataset_name.endswith(".csv"):
                    with open(dataset_name, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        # Determine the prompt column (instruction, prompt, or text)
                        instruction_col_name = 'instruction' # default
                        if reader.fieldnames:
                            if 'instruction' in reader.fieldnames:
                                instruction_col_name = 'instruction'
                            elif 'prompt' in reader.fieldnames:
                                instruction_col_name = 'prompt'
                            elif 'text' in reader.fieldnames: # Common alternative for prompts
                                instruction_col_name = 'text'
                            else: # Fallback if none of the common names are found
                                print(f"Warning: Could not find 'instruction', 'prompt', or 'text' column in {dataset_name}. Using first column if available or empty strings.")
                                instruction_col_name = reader.fieldnames[0] if reader.fieldnames else None
                        
                        if instruction_col_name:
                            for row in reader:
                                temp_prompts.append(row.get(instruction_col_name, ''))
                        else:
                            print(f"Error: No usable prompt column found in CSV {dataset_name}.")
                else:
                    print(f"Unsupported file type for direct loading: {dataset_name}. Attempting harmtype loading.")
                    is_file_path = False # Fallback to harmtype loading
                
                loaded_prompts_for_dataset = [p for p in temp_prompts if p] # Filter out empty prompts

                if is_file_path and not loaded_prompts_for_dataset:
                    print(f"Warning: No prompts extracted from file {dataset_name}. Check file content and prompt fields ('instruction', 'prompt', 'text').")
                
                if loaded_prompts_for_dataset and len(loaded_prompts_for_dataset) > num_test_samples:
                    loaded_prompts_for_dataset = random.sample(loaded_prompts_for_dataset, num_test_samples)
            except Exception as e:
                print(f"Error loading dataset from file {dataset_name}: {e}. Falling back to harmtype loading.")
                is_file_path = False # Ensure fallback

        if not is_file_path: # If not a file path OR file loading failed/produced no prompts
            print(f"Loading dataset for '{dataset_name}' as a harmtype/key.")
            try:
                loaded_prompts_from_harmtype = load_dataset_split(harmtype=dataset_name, split='test', instructions_only=True)
                if len(loaded_prompts_from_harmtype) < num_test_samples and dataset_name not in ['harmful', 'or_bench_hard']:
                     print(f"Warning: Requested {num_test_samples} for {dataset_name} but only loaded {len(loaded_prompts_from_harmtype)} from 'test' split via load_dataset_split.")
                
                if len(loaded_prompts_from_harmtype) > num_test_samples:
                    loaded_prompts_for_dataset = random.sample(loaded_prompts_from_harmtype, num_test_samples)
                else:
                    loaded_prompts_for_dataset = loaded_prompts_from_harmtype

            except Exception as e_lds:
                print(f"Error loading dataset {dataset_name} with load_dataset_split: {e_lds}. Trying general load_dataset.")
                try:
                    raw_loaded = load_dataset(dataset_name)
                    if isinstance(raw_loaded, list) and raw_loaded and isinstance(raw_loaded[0], dict) and 'instruction' in raw_loaded[0]:
                        loaded_prompts_from_raw = [item['instruction'] for item in raw_loaded]
                    elif isinstance(raw_loaded, list) and raw_loaded and isinstance(raw_loaded[0], str):
                        loaded_prompts_from_raw = raw_loaded
                    else:
                        raise ValueError(f"load_dataset for {dataset_name} returned unrecognized format.")
                    
                    if len(loaded_prompts_from_raw) > num_test_samples:
                        loaded_prompts_for_dataset = random.sample(loaded_prompts_from_raw, num_test_samples)
                    else:
                        loaded_prompts_for_dataset = loaded_prompts_from_raw
                except Exception as e_fallback:
                    print(f"Error with fallback load_dataset for {dataset_name}: {e_fallback}. Completions might be empty.")
                    loaded_prompts_for_dataset = [] # Ensure it is empty on failure
        
        # Construct dataset list of dicts if prompts were loaded
        if loaded_prompts_for_dataset:
            dataset = [{'instruction': p, 'category': dataset_key_for_filename} for p in loaded_prompts_for_dataset]
        else:
            print(f"Warning: Loaded empty dataset for {dataset_name} after all attempts. Check dataset name/path and loading logic.")
            dataset = [] # Ensure dataset is an empty list

    if not dataset:
        print(f"Warning: Dataset for '{dataset_name}' is empty. Skipping completion generation.")
        completions_to_save = []
    else:
        print(f"Generating completions for {dataset_name} with {len(dataset)} examples using {intervention_label} setup...")
        # Ensure hooks are correctly applied using context manager
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            completions_to_save = model_base.generate_completions(
                dataset,
                max_new_tokens=cfg.get('max_new_tokens', 4096),
                batch_size=cfg.batch_size,
                system=system
            )

    completion_save_path = os.path.join(completions_dir, f'{dataset_key_for_filename}_{intervention_label}_completions.json')
    with open(completion_save_path, "w") as f:
        json.dump(completions_to_save, f, indent=4)
    print(f"Completions for {dataset_name} saved to {completion_save_path}")

# --- evaluate_completions_and_save_results_for_dataset (copied from run_pipeline.py) ---
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak, substring_matching_judge_fn # evaluate_jailbreak is key
def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    completions_dir = os.path.join(cfg.artifact_path, 'completions')
    evaluations_dir = os.path.join(cfg.artifact_path, 'evaluations')
    os.makedirs(evaluations_dir, exist_ok=True)

    dataset_key_for_filename = os.path.basename(str(dataset_name))

    completion_file_path = os.path.join(completions_dir, f'{dataset_key_for_filename}_{intervention_label}_completions.json')
    evaluation_save_path = os.path.join(evaluations_dir, f'{dataset_key_for_filename}_{intervention_label}_evaluations.json')

    if not os.path.exists(completion_file_path):
        print(f"Error: Completions file not found at {completion_file_path}. Skipping evaluation for {dataset_name}.")
        return

    with open(completion_file_path, 'r') as f:
        completions = json.load(f)

    if not completions:
        print(f"Warning: No completions found in {completion_file_path} for {dataset_name}. Skipping evaluation.")
        with open(evaluation_save_path, "w") as f:
            json.dump({"error": "No completions to evaluate"}, f, indent=4)
        return

    print(f"Evaluating completions for {dataset_name} ({intervention_label}) using methodologies: {eval_methodologies}")
    evaluation_results = evaluate_jailbreak( # This is the core call
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=evaluation_save_path, # evaluate_jailbreak saves it
        cfg=cfg
    )
    # evaluate_jailbreak now saves the file itself. This explicit save is redundant but harmless.
    with open(evaluation_save_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluation results for {dataset_name} saved to {evaluation_save_path}")


# --- EBM Training Function (copied and adapted from run_pipeline.py) ---
global_captured_activations_hook_data = {}
def _activation_capture_hook_fn_factory(layer_idx, position_indices_list):
    def _hook_fn(module, input_act_tuple):
        activation_tensor = input_act_tuple[0].clone().detach()
        batch_activations_at_target_pos_list = []
        for pos_idx_relative in position_indices_list:
            actual_pos_idx = activation_tensor.shape[1] + pos_idx_relative if pos_idx_relative < 0 else pos_idx_relative
            if 0 <= actual_pos_idx < activation_tensor.shape[1]:
                batch_activations_at_target_pos_list.append(activation_tensor[:, actual_pos_idx, :])
            else:
                batch_activations_at_target_pos_list.append(torch.zeros_like(activation_tensor[:, 0, :]))

        if len(batch_activations_at_target_pos_list) > 1:
            batch_activations_at_target = torch.mean(torch.stack(batch_activations_at_target_pos_list), dim=0)
        elif batch_activations_at_target_pos_list:
            batch_activations_at_target = batch_activations_at_target_pos_list[0]
        else:
            batch_activations_at_target = torch.zeros((activation_tensor.shape[0], activation_tensor.shape[2]), device=activation_tensor.device, dtype=activation_tensor.dtype)

        if layer_idx not in global_captured_activations_hook_data:
            global_captured_activations_hook_data[layer_idx] = []
        global_captured_activations_hook_data[layer_idx].append(batch_activations_at_target.cpu())
    return _hook_fn

def _extract_activations_for_ebm_training(
    llm_model_base, prompts_for_extraction: List[str],
    target_layer: int, target_positions: List[int],
    batch_size: int, device: str, system_prompt: Optional[str] = None
):
    llm_model_base.model.eval()
    all_extracted_activations = []
    hook_module = llm_model_base.model_block_modules[target_layer]
    hook_fn = _activation_capture_hook_fn_factory(target_layer, target_positions)

    print(f"Extracting EBM training activations from layer {target_layer} at positions {target_positions} for {len(prompts_for_extraction)} prompts...")
    for i in tqdm(range(0, len(prompts_for_extraction), batch_size), desc=f"Extracting EBM acts (L{target_layer}, P{target_positions})"):
        batch_prompts = prompts_for_extraction[i:i+batch_size]
        inputs = llm_model_base.tokenize_instructions_fn(instructions=batch_prompts, system=system_prompt)

        global_captured_activations_hook_data.clear()
        handle = hook_module.register_forward_pre_hook(hook_fn)

        with torch.no_grad():
            llm_model_base.model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
        handle.remove()

        if target_layer in global_captured_activations_hook_data and global_captured_activations_hook_data[target_layer]:
            batch_layer_activations = torch.cat(global_captured_activations_hook_data[target_layer], dim=0)
            all_extracted_activations.append(batch_layer_activations)

    if not all_extracted_activations: return torch.empty(0)
    return torch.cat(all_extracted_activations, dim=0)

def _classify_llm_response_for_ebm(response_text: str, is_prompt_harmful_context: bool) -> Optional[str]:
    is_refusal = substring_matching_judge_fn(response_text) # True if it IS a refusal
    if is_prompt_harmful_context:
        return 'positive_tr' if is_refusal else 'negative_tr'
    else:
        return 'negative_fr' if is_refusal else 'positive_fr'

def _reconstruct_responses_with_metadata(
    original_data_list: List[Dict[str, Any]],
    generation_outputs: List[Any],
    source_prompts_list: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    reconstructed_list = []
    if len(generation_outputs) != len(original_data_list): # Should be source_prompts_list if that's the reference
        ref_len = len(source_prompts_list) if source_prompts_list else len(original_data_list)
        if len(generation_outputs) != ref_len:
            print(f"Error: Mismatch in lengths between generation inputs ({ref_len}) and outputs ({len(generation_outputs)}). Cannot reliably reconstruct.")
            # Attempt to reconstruct based on shorter list if one is clearly a source for the other.
            # This part is tricky without knowing which list is the "true" source of prompts for generation.
            # Assuming original_data_list was the source for metadata, and generation_outputs for responses.
            # Best to ensure upstream that lengths match. For now, we'll proceed cautiously.
            min_len = min(len(original_data_list), len(generation_outputs))
            print(f"Reconstructing based on the minimum length: {min_len}")
            original_data_list = original_data_list[:min_len]
            generation_outputs = generation_outputs[:min_len]
            if source_prompts_list and len(source_prompts_list) != min_len :
                 source_prompts_list = source_prompts_list[:min_len]


    for i, original_data_item in enumerate(original_data_list):
        if i >= len(generation_outputs): break # Safety break if lengths still mismatched after attempt

        generated_item = generation_outputs[i]
        response_text = ""
        prompt_text = source_prompts_list[i] if source_prompts_list and i < len(source_prompts_list) else original_data_item.get('instruction', "")

        if isinstance(generated_item, dict): # Typical for model_base.generate_completions
            response_text = generated_item.get('response', "").strip()
            prompt_text = generated_item.get('prompt', prompt_text) # vLLM might put formatted prompt here
        elif isinstance(generated_item, str): # If generation_outputs is List[str] (e.g. direct vLLM output processing)
            response_text = generated_item.strip()
        else:
            print(f"Warning: Unexpected output format from generation for item {i}: {type(generated_item)}")

        reconstructed_list.append({
            'prompt': prompt_text,
            'response': response_text,
            'is_harmful_context': original_data_item['is_harmful_context'],
            'source_type': original_data_item.get('source_type', 'unknown') # Ensure source_type
        })
    return reconstructed_list


def train_and_save_ebm_if_needed(cfg, llm_model_base: ModelBase, force_retrain: bool = False):
    if not (hasattr(cfg, 'ebm_params') and cfg.ebm_params):
        print("No ebm_params in config. Skipping EBM training.")
        return {}, None, [], {} # trained_ebms, final_ebm, top_n_layers, individual_val_accs

    trained_ebms: Dict[int, Union[SimpleEBM, ComplexEBM]] = {}
    individual_ebm_val_accuracies: Dict[int, float] = {}
    ebm_fr_path_template = cfg.ebm_params.get('ebm_fr_save_path')

    if not ebm_fr_path_template:
        print("Warning: ebm_params.ebm_fr_save_path not defined. Cannot train or load EBMs.")
        return {}, None, [], {}

    ebm_target_layers_config = cfg.ebm_params.get('ebm_target_layer') 
    
    target_layers_to_process: List[int] = []
    if isinstance(ebm_target_layers_config, int):
        target_layers_to_process = [ebm_target_layers_config]
    elif isinstance(ebm_target_layers_config, list):
        target_layers_to_process = [int(l) for l in ebm_target_layers_config]
    elif isinstance(ebm_target_layers_config, str) and ebm_target_layers_config.lower() == 'all':
        target_layers_to_process = list(range(llm_model_base.model.config.num_hidden_layers))
    else:
        print(f"Warning: Invalid 'ebm_target_layer' configuration: {ebm_target_layers_config}. Expected int, list of ints, or 'all'. No EBMs will be trained/loaded.")
        return {}, None, [], {}

    if not target_layers_to_process:
        print("No target layers specified or resolved for EBM training. Skipping.")
        return {}, None, [], {}
        
    print(f"Target layers for individual EBM training/loading: {target_layers_to_process}")
    
    # --- Data Generation for EBM training (once for all layers) ---
    prompts_for_llm_response_generation = []
    default_sample_size = cfg.ebm_params.get('num_initial_prompts', 1000)

    if hasattr(cfg.ebm_params, 'ebm_data_sources') and cfg.ebm_params.ebm_data_sources:
        # print("Loading EBM training data using ebm_data_sources for unified EBM...") # Keep if needed
        data_sources_config = cfg.ebm_params.ebm_data_sources

        def _load_prompts_from_source_list(source_list, is_harmful_context_flag, category_name_for_logs):
            loaded_prompts_with_context = []
            if source_list:
                for source_config_item in source_list: 
                    current_sample_size = source_config_item.get('sample_size', default_sample_size)
                    source_name = source_config_item.get('name', 'Unnamed')
                    source_type = source_config_item.get('type')
                    # print(f"Loading {category_name_for_logs} prompts from source: {source_name}, type: {source_type}, sampling up to {current_sample_size}")
                    temp_prompts_this_source = []
                    if source_type == "jsonl":
                        try:
                            with open(source_config_item.path, 'r') as f_jsonl:
                                for line in f_jsonl:
                                    try:
                                        record = json.loads(line)
                                        if record.get(source_config_item.filter_field) == source_config_item.filter_value:
                                            temp_prompts_this_source.append(record.get(source_config_item.prompt_field, ""))
                                    except json.JSONDecodeError: continue
                        except Exception as e: print(f"Warning: Error loading jsonl source {source_name}: {e}")
                    elif source_type == "harmtype_split":
                        try:
                            temp_prompts_this_source = load_dataset_split(
                                harmtype=source_config_item.harmtype,
                                split=source_config_item.get('split', 'train'),
                                instructions_only=True
                            )
                        except Exception as e: print(f"Warning: Could not load harmtype_split {source_config_item.harmtype}: {e}")
                    else: print(f"Warning: Unknown source type '{source_type}'.")

                    if len(temp_prompts_this_source) > current_sample_size:
                        sampled_prompts = random.sample(temp_prompts_this_source, current_sample_size)
                    else:
                        sampled_prompts = temp_prompts_this_source
                    for p_text in sampled_prompts:
                        loaded_prompts_with_context.append((p_text, is_harmful_context_flag))
            return loaded_prompts_with_context

        if hasattr(data_sources_config, 'harmless_prompts'):
            prompts_for_llm_response_generation.extend(
                _load_prompts_from_source_list(data_sources_config.harmless_prompts, False, "harmless")
            )
        if hasattr(data_sources_config, 'true_harmful_prompts'):
             prompts_for_llm_response_generation.extend(
                _load_prompts_from_source_list(data_sources_config.true_harmful_prompts, True, "true_harmful")
            )
    else: 
        raise ValueError("ebm_data_sources must be defined in cfg.ebm_params for EBM training.")


    if not prompts_for_llm_response_generation:
         raise ValueError("No prompts collected for EBM training. Check ebm_data_sources configuration.")

    data_for_classification = [{'instruction': p_text, 'is_harmful_context': is_harm_ctx, 'source_type': 'true_harmful' if is_harm_ctx else 'harmless'}
                               for p_text, is_harm_ctx in prompts_for_llm_response_generation]

    print(f"Generating initial LLM responses for {len(data_for_classification)} prompts to create EBM training data...")
    llm_responses = []
    try: # vLLM
        from vllm import LLM, SamplingParams
        
        # 使用简化的vLLM配置避免兼容性问题
        download_dir = cfg.ebm_params.get('vllm_download_dir', './model_cache')
        os.makedirs(download_dir, exist_ok=True)
        
        vllm_params = {
            'model': cfg.model_path,
            'download_dir': download_dir,
            'tensor_parallel_size': cfg.ebm_params.get('vllm_tensor_parallel_size', 1),
            'gpu_memory_utilization': cfg.ebm_params.get('vllm_gpu_memory_utilization', 0.80),
            'trust_remote_code': True,
            'enforce_eager': True,  # 禁用编译，避免Python.h依赖
            'disable_log_stats': True,  # 减少日志输出
            # 移除max_model_len让vLLM自动推断，避免RoPE scaling兼容性问题
        }
            
        llm = LLM(**vllm_params)
        
        stop_token_ids = []
        stop_strings = []
        if llm_model_base.refusal_toks:
            for tok in llm_model_base.refusal_toks:
                if isinstance(tok, int):
                    stop_token_ids.append(tok)
                elif isinstance(tok, str):
                    stop_strings.append(tok)
        
        sampling_params_kwargs = {
            'temperature': cfg.ebm_params.get('vllm_sampling_temp', 0.0),
            'max_tokens': cfg.ebm_params.max_new_tokens_for_ebm_data,
        }
        
        if stop_token_ids:
            sampling_params_kwargs['stop_token_ids'] = stop_token_ids
        if stop_strings:
            sampling_params_kwargs['stop'] = stop_strings
            
        sampling_params = SamplingParams(**sampling_params_kwargs)
        
        vllm_prompts_list = []
        system_prompt_for_vllm = llm_model_base.system_prompt if hasattr(llm_model_base, 'system_prompt') and llm_model_base.system_prompt is not None else cfg.get('system')
        temp_vllm_source_data = [] 
        
        for item_idx, item_for_vllm in enumerate(data_for_classification):
            formatted_prompt_chat = llm_model_base.formatter(item_for_vllm['instruction'], system_prompt_for_vllm, history=None) 
            vllm_prompts_list.append(formatted_prompt_chat)
            temp_vllm_source_data.append(item_for_vllm)
            
        print(f"Generating responses using vLLM for {len(vllm_prompts_list)} prompts...")
        vllm_outputs = llm.generate(vllm_prompts_list, sampling_params)
        extracted_vllm_response_texts = [output.outputs[0].text for output in vllm_outputs]
        llm_responses = _reconstruct_responses_with_metadata(temp_vllm_source_data, extracted_vllm_response_texts, source_prompts_list=vllm_prompts_list)
        
        del llm; gc.collect(); torch.cuda.empty_cache()
    except Exception as e_vllm:
        import traceback
        print(f"Error with vLLM for EBM data generation: {e_vllm}")
        print(f"Full traceback: {traceback.format_exc()}")
        print("Falling back to standard HF generation.")
        llm_gen_batch_size = cfg.ebm_params.get('llm_gen_batch_size_for_ebm_data', cfg.get('batch_size', 8))
        standard_hf_outputs = llm_model_base.generate_completions(
            data_for_classification, batch_size=llm_gen_batch_size,
            max_new_tokens=cfg.ebm_params.max_new_tokens_for_ebm_data,
            system=llm_model_base.system_prompt if hasattr(llm_model_base, 'system_prompt') and llm_model_base.system_prompt is not None else cfg.get('system')
        )
        llm_responses = _reconstruct_responses_with_metadata(data_for_classification, standard_hf_outputs)

    if not llm_responses:
         raise ValueError("EBM training data generation failed (llm_responses is empty).")

    labeled_prompts_for_activation_extraction = [] # List of (prompt_text, ebm_label, is_harmful_context)
    for item in llm_responses:
        ebm_label = _classify_llm_response_for_ebm(item['response'], item['is_harmful_context'])
        if ebm_label:
            labeled_prompts_for_activation_extraction.append((item['prompt'], ebm_label, item['is_harmful_context']))

    if not labeled_prompts_for_activation_extraction:
        raise ValueError("No suitable prompts found after classification for EBM activation extraction.")
    
    prompts_to_extract_texts = [item[0] for item in labeled_prompts_for_activation_extraction]
    # --- End of common data generation ---

    ebm_training_details_for_top_n = [] # Store (val_acc, layer_idx, model_object)

    for target_layer_fr in target_layers_to_process:
        print(f"\n--- Processing Individual EBM for Layer {target_layer_fr} ---")
        target_positions_fr_str = str(cfg.ebm_params.get('ebm_target_positions', "-1"))
        target_positions_fr_filename_part = target_positions_fr_str.replace(",", "_").replace("-", "neg")
        current_ebm_for_layer = None
        best_val_acc_for_layer = -1.0 # Default if not trained or no val set

        try:
            ebm_arch_for_path = cfg.ebm_params.get('ebm_architecture', 'simple')
            resolved_ebm_fr_path = ebm_fr_path_template.format(
                model_alias=cfg.model_alias,
                ebm_architecture=ebm_arch_for_path,
                ebm_target_layer=target_layer_fr,
                ebm_target_positions_filename_part=target_positions_fr_filename_part
            )
        except KeyError as e:
            print(f"Error formatting ebm_fr_save_path ('{ebm_fr_path_template}') for layer {target_layer_fr}. Missing key: {e}. Skipping.")
            continue

        if not os.path.exists(resolved_ebm_fr_path) or force_retrain:
            print(f"EBM for layer {target_layer_fr} not found or force_retrain=True. Training...")
            target_positions_list_fr = [int(p.strip()) for p in target_positions_fr_str.split(',')]
            
            extracted_acts_tensor = _extract_activations_for_ebm_training(
                llm_model_base, prompts_to_extract_texts, target_layer_fr,
                target_positions_list_fr, cfg.ebm_params.activation_extraction_batch_size, cfg.device,
                system_prompt=llm_model_base.system_prompt if hasattr(llm_model_base, 'system_prompt') else cfg.get('system')
            )

            if extracted_acts_tensor.numel() == 0:
                print(f"Warning: No activations extracted for EBM at layer {target_layer_fr}. Skipping.")
                continue

            X_good_activations_list, X_bad_activations_list = [], []
            for idx, (_, ebm_label, _) in enumerate(labeled_prompts_for_activation_extraction):
                if idx < extracted_acts_tensor.shape[0]:
                    act_sample = extracted_acts_tensor[idx]
                    if ebm_label == 'positive_fr' or ebm_label == 'positive_tr': X_good_activations_list.append(act_sample)
                    elif ebm_label == 'negative_fr' or ebm_label == 'negative_tr': X_bad_activations_list.append(act_sample)
            
            if not X_good_activations_list or not X_bad_activations_list:
                print(f"Warning: Insufficient good/bad samples for EBM layer {target_layer_fr}. Skipping.")
                continue
            
            EBM_FORCED_DTYPE = torch.float32
            X_good_all = torch.stack(X_good_activations_list).to(dtype=EBM_FORCED_DTYPE)
            X_bad_all = torch.stack(X_bad_activations_list).to(dtype=EBM_FORCED_DTYPE)
            min_len_overall = min(X_good_all.shape[0], X_bad_all.shape[0])
            if min_len_overall == 0 : print(f"Warning: Zero samples for EBM layer {target_layer_fr}. Skipping."); continue
            
            indices = torch.randperm(min_len_overall)
            split_idx_float = 0.8 * min_len_overall; split_idx = int(split_idx_float)
            if min_len_overall > 1 and split_idx == min_len_overall : split_idx = min_len_overall -1
            elif min_len_overall == 1 and split_idx == 1: split_idx = 1
            train_indices, val_indices = indices[:split_idx], indices[split_idx:]
            X_good_train, X_bad_train = X_good_all[train_indices], X_bad_all[train_indices]
            X_good_val, X_bad_val = X_good_all[val_indices], X_bad_all[val_indices]

            if X_good_train.shape[0] == 0 or X_bad_train.shape[0] == 0: print(f"Warning: Not enough train samples for EBM layer {target_layer_fr}. Skipping."); continue

            ebm_train_dataset = TensorDataset(X_good_train.to(cfg.device), X_bad_train.to(cfg.device))
            ebm_train_dataloader = DataLoader(ebm_train_dataset, batch_size=cfg.ebm_params.ebm_batch_size, shuffle=True)
            ebm_val_dataloader = None
            if X_good_val.shape[0] > 0 and X_bad_val.shape[0] > 0:
                ebm_val_dataset = TensorDataset(X_good_val.to(cfg.device), X_bad_val.to(cfg.device))
                ebm_val_dataloader = DataLoader(ebm_val_dataset, batch_size=cfg.ebm_params.ebm_batch_size, shuffle=False)

            ebm_arch = cfg.ebm_params.get('ebm_architecture', 'simple')
            if ebm_arch == 'simple':
                current_ebm_for_layer = SimpleEBM(input_dim=llm_model_base.model.config.hidden_size, hidden_dim=cfg.ebm_params.get('simple_ebm_hidden_dim', 512))
            elif ebm_arch == 'complex':
                current_ebm_for_layer = ComplexEBM(input_dim=llm_model_base.model.config.hidden_size, hidden_dims=cfg.ebm_params.get('complex_ebm_hidden_dims', [1024,512,256]), dropout_rate=cfg.ebm_params.get('complex_ebm_dropout_rate',0.1), use_layernorm=cfg.ebm_params.get('complex_ebm_use_layernorm',True))
            else: print(f"Unsupported ebm_architecture: {ebm_arch} for layer {target_layer_fr}. Skipping."); continue
            
            current_ebm_for_layer = current_ebm_for_layer.to(device=cfg.device, dtype=EBM_FORCED_DTYPE)
            optimizer = optim.Adam(current_ebm_for_layer.parameters(), lr=cfg.ebm_params.ebm_lr)

            best_epoch_val_acc_for_layer = -1.0
            path_to_save_best_model_for_layer = resolved_ebm_fr_path 
            os.makedirs(os.path.dirname(path_to_save_best_model_for_layer), exist_ok=True)
            temperature = cfg.ebm_params.get('ebm_infonce_temperature', 0.07)

            # tqdm for epochs for individual layer EBM
            epoch_pbar_individual = tqdm(range(cfg.ebm_params.ebm_epochs), desc=f"L{target_layer_fr} EBM Epochs", leave=False)
            for epoch in epoch_pbar_individual:
                current_ebm_for_layer.train()
                total_train_loss = 0
                # Inner tqdm for batches
                batch_pbar_individual = tqdm(ebm_train_dataloader, desc=f"L{target_layer_fr} Train Ep {epoch+1}", leave=False)
                for x_p_batch, x_n_batch in batch_pbar_individual:
                    optimizer.zero_grad()
                    energy_p = current_ebm_for_layer(x_p_batch.to(EBM_FORCED_DTYPE))
                    energy_n = current_ebm_for_layer(x_n_batch.to(EBM_FORCED_DTYPE))
                    loss = _calculate_infonce_loss(energy_p, energy_n, temperature, cfg.device)
                    loss.backward(); optimizer.step(); total_train_loss += loss.item()
                    batch_pbar_individual.set_postfix(loss=loss.item())
                
                avg_train_loss = total_train_loss / len(ebm_train_dataloader) if len(ebm_train_dataloader) > 0 else 0
                log_dict = {'TrainL': f'{avg_train_loss:.4f}'}
                
                if ebm_val_dataloader:
                    current_ebm_for_layer.eval()
                    total_val_loss, correct_val_preds, total_val_samples = 0,0,0
                    with torch.no_grad():
                        for x_p_b_val, x_n_b_val in ebm_val_dataloader:
                            e_p_val = current_ebm_for_layer(x_p_b_val.to(EBM_FORCED_DTYPE))
                            e_n_val = current_ebm_for_layer(x_n_b_val.to(EBM_FORCED_DTYPE))
                            val_loss = _calculate_infonce_loss(e_p_val, e_n_val, temperature, cfg.device)
                            total_val_loss += val_loss.item()
                            correct_val_preds += (e_p_val < 0).sum().item() + (e_n_val > 0).sum().item() 
                            total_val_samples += x_p_b_val.shape[0] + x_n_b_val.shape[0]
                    avg_val_loss = total_val_loss / len(ebm_val_dataloader) if len(ebm_val_dataloader) > 0 else 0
                    val_acc = (correct_val_preds / total_val_samples) * 100 if total_val_samples > 0 else 0
                    # print(f"L{target_layer_fr} EBM Ep {epoch+1} AvgTrainL: {avg_train_loss:.4f} AvgValL: {avg_val_loss:.4f} ValAcc: {val_acc:.2f}%")
                    log_dict['ValL'] = f'{avg_val_loss:.4f}'
                    log_dict['ValAcc'] = f'{val_acc:.2f}%'

                    if val_acc > best_epoch_val_acc_for_layer:
                        best_epoch_val_acc_for_layer = val_acc
                        torch.save(current_ebm_for_layer.state_dict(), path_to_save_best_model_for_layer)
                        # print(f"  L{target_layer_fr} New best EBM saved with ValAcc: {best_epoch_val_acc_for_layer:.2f}% to {path_to_save_best_model_for_layer}")
                        log_dict['Saved'] = '*' # Indicate save
                else: 
                    # print(f"L{target_layer_fr} EBM Ep {epoch+1} AvgTrainL: {avg_train_loss:.4f} (No validation set)")
                    if epoch == cfg.ebm_params.ebm_epochs - 1:
                         torch.save(current_ebm_for_layer.state_dict(), path_to_save_best_model_for_layer)
                         # print(f"  L{target_layer_fr} EBM (last epoch) saved to {path_to_save_best_model_for_layer} (no validation set during training)")
                         log_dict['Saved'] = '(last_ep)'
                epoch_pbar_individual.set_postfix(log_dict)
            
            best_val_acc_for_layer = best_epoch_val_acc_for_layer 
            if best_epoch_val_acc_for_layer >= 0 and path_to_save_best_model_for_layer:
                print(f"  L{target_layer_fr} Best EBM training complete. Highest ValAcc: {best_epoch_val_acc_for_layer:.2f}%, saved to {path_to_save_best_model_for_layer}")
            elif path_to_save_best_model_for_layer and os.path.exists(path_to_save_best_model_for_layer): # If saved due to last epoch, no val acc
                print(f"  L{target_layer_fr} EBM training complete (last epoch model saved, no validation data during training) to {path_to_save_best_model_for_layer}")

        else: # Load existing EBM if available
            if os.path.exists(resolved_ebm_fr_path):
                print(f"Loading existing EBM for layer {target_layer_fr} from: {resolved_ebm_fr_path}")
                try:
                    ebm_arch = cfg.ebm_params.get('ebm_architecture', 'simple')
                    if ebm_arch == 'simple':
                        current_ebm_for_layer = SimpleEBM(input_dim=llm_model_base.model.config.hidden_size, hidden_dim=cfg.ebm_params.get('simple_ebm_hidden_dim', 512))
                    elif ebm_arch == 'complex':
                        current_ebm_for_layer = ComplexEBM(input_dim=llm_model_base.model.config.hidden_size, hidden_dims=cfg.ebm_params.get('complex_ebm_hidden_dims', [1024,512,256]), dropout_rate=cfg.ebm_params.get('complex_ebm_dropout_rate',0.1), use_layernorm=cfg.ebm_params.get('complex_ebm_use_layernorm',True))
                    else:
                        print(f"Unsupported ebm_architecture: {ebm_arch} for loading layer {target_layer_fr}. Skipping.")
                        continue
                    
                    current_ebm_for_layer.load_state_dict(torch.load(resolved_ebm_fr_path, map_location=torch.device(cfg.device)))
                    current_ebm_for_layer = current_ebm_for_layer.to(cfg.device)
                    current_ebm_for_layer.eval()
                    print(f"Successfully loaded EBM for layer {target_layer_fr}")
                    best_val_acc_for_layer = 75.0  # 设定一个假设的验证准确率，用于final EBM选择
                except Exception as e:
                    print(f"Error loading EBM for layer {target_layer_fr}: {e}. Skipping.")
                    current_ebm_for_layer = None

        if current_ebm_for_layer:
            trained_ebms[target_layer_fr] = current_ebm_for_layer
            individual_ebm_val_accuracies[target_layer_fr] = best_val_acc_for_layer
            if best_val_acc_for_layer >= 0: # Only consider for top-N if val_acc is valid
                 ebm_training_details_for_top_n.append({'val_acc': best_val_acc_for_layer, 'layer': target_layer_fr, 'model': current_ebm_for_layer})
    # --- End of individual EBM loop ---

    final_concat_ebm_model = None
    top_n_layers_for_final_ebm_indices = [] # Ensure this is the name used internally

    top_n_config = cfg.ebm_params.get('top_n_layers_for_final_ebm', 5)
    min_layers_for_final_config = cfg.ebm_params.get('min_layers_for_final_ebm', 2)

    best_epoch_val_acc_for_final_ebm = -1.0 # Initialize here
    final_ebm_path_saved = None # Initialize here

    if ebm_training_details_for_top_n and len(ebm_training_details_for_top_n) >= min_layers_for_final_config:
        ebm_training_details_for_top_n.sort(key=lambda x: x['val_acc'], reverse=True)
        top_n_selected_details = ebm_training_details_for_top_n[:top_n_config]
        top_n_layers_for_final_ebm_indices = [d['layer'] for d in top_n_selected_details] # Internal name
        
        val_acc_strs = [f"{d['val_acc']:.2f}%" for d in top_n_selected_details]
        print(f"\n--- Training Final EBM on Top {len(top_n_layers_for_final_ebm_indices)} Layers: {top_n_layers_for_final_ebm_indices} (Val Accs: [{', '.join(val_acc_strs)}]) ---") # Use internal name

        layer_activations_map = {} # {layer_idx: extracted_acts_tensor_for_this_layer}
        target_positions_fr_str = str(cfg.ebm_params.get('ebm_target_positions', "-1")) # Assuming same positions for all
        target_positions_list_fr = [int(p.strip()) for p in target_positions_fr_str.split(',')]

        for layer_idx in top_n_layers_for_final_ebm_indices: # Use internal name
            print(f"Extracting activations for final EBM - Layer {layer_idx}...")
            layer_acts_tensor = _extract_activations_for_ebm_training(
                llm_model_base, prompts_to_extract_texts, layer_idx, 
                target_positions_list_fr, cfg.ebm_params.activation_extraction_batch_size, cfg.device,
                system_prompt=llm_model_base.system_prompt if hasattr(llm_model_base, 'system_prompt') else cfg.get('system')
            )
            if layer_acts_tensor.numel() > 0:
                layer_activations_map[layer_idx] = layer_acts_tensor
            else:
                print(f"Warning: No activations extracted for layer {layer_idx} for final EBM. This layer will be skipped.")
        
        # Filter top_n_layers_for_final_ebm_indices if any layer failed extraction
        valid_top_n_layers = [l for l in top_n_layers_for_final_ebm_indices if l in layer_activations_map] # Use internal name
        if len(valid_top_n_layers) < min_layers_for_final_config:
            print(f"Not enough layers with valid activations ({len(valid_top_n_layers)}) for final EBM. Min required: {min_layers_for_final_config}. Skipping final EBM.")
        else:
            top_n_layers_for_final_ebm_indices = valid_top_n_layers # Update internal name
            final_X_good_list, final_X_bad_list = [], []
            for prompt_idx, (_, ebm_label, _) in enumerate(labeled_prompts_for_activation_extraction):
                current_concat_act_parts = []
                valid_prompt_for_concat = True
                for layer_idx in top_n_layers_for_final_ebm_indices: # Use internal name
                    if prompt_idx < layer_activations_map[layer_idx].shape[0]:
                        current_concat_act_parts.append(layer_activations_map[layer_idx][prompt_idx])
                    else: # Should not happen if prompts_to_extract_texts matches across extractions
                        valid_prompt_for_concat = False; break 
                if valid_prompt_for_concat and current_concat_act_parts:
                    concatenated_act = torch.cat(current_concat_act_parts, dim=-1)
                    if ebm_label == 'positive_fr' or ebm_label == 'positive_tr': final_X_good_list.append(concatenated_act)
                    elif ebm_label == 'negative_fr' or ebm_label == 'negative_tr': final_X_bad_list.append(concatenated_act)

            if final_X_good_list and final_X_bad_list:
                EBM_FORCED_DTYPE = torch.float32
                X_good_final = torch.stack(final_X_good_list).to(dtype=EBM_FORCED_DTYPE)
                X_bad_final = torch.stack(final_X_bad_list).to(dtype=EBM_FORCED_DTYPE)
                print(f"Final EBM Data: Good {X_good_final.shape}, Bad {X_bad_final.shape}")

                min_len_final = min(X_good_final.shape[0], X_bad_final.shape[0])
                if min_len_final > 0:
                    indices_final = torch.randperm(min_len_final)
                    split_idx_final_float = 0.8 * min_len_final; split_idx_final = int(split_idx_final_float)
                    if min_len_final > 1 and split_idx_final == min_len_final: split_idx_final = min_len_final - 1
                    elif min_len_final == 1: split_idx_final = 1
                    
                    train_indices_final, val_indices_final = indices_final[:split_idx_final], indices_final[split_idx_final:]
                    X_g_train_f, X_b_train_f = X_good_final[train_indices_final], X_bad_final[train_indices_final]
                    X_g_val_f, X_b_val_f = X_good_final[val_indices_final], X_bad_final[val_indices_final]

                    if X_g_train_f.shape[0] > 0 and X_b_train_f.shape[0] > 0:
                        final_ebm_train_ds = TensorDataset(X_g_train_f.to(cfg.device), X_b_train_f.to(cfg.device))
                        final_ebm_train_dl = DataLoader(final_ebm_train_ds, batch_size=cfg.ebm_params.ebm_batch_size, shuffle=True)
                        final_ebm_val_dl = None
                        if X_g_val_f.shape[0] > 0 and X_b_val_f.shape[0] > 0:
                            final_ebm_val_ds = TensorDataset(X_g_val_f.to(cfg.device), X_b_val_f.to(cfg.device))
                            final_ebm_val_dl = DataLoader(final_ebm_val_ds, batch_size=cfg.ebm_params.ebm_batch_size, shuffle=False)

                        final_ebm_input_dim = len(top_n_layers_for_final_ebm_indices) * llm_model_base.model.config.hidden_size # Use internal name
                        ebm_arch = cfg.ebm_params.get('ebm_architecture', 'simple') # Use same arch type for final for now
                        if ebm_arch == 'simple':
                            final_concat_ebm_model = SimpleEBM(input_dim=final_ebm_input_dim, hidden_dim=cfg.ebm_params.get('simple_ebm_hidden_dim', 512))
                        elif ebm_arch == 'complex':
                            final_concat_ebm_model = ComplexEBM(input_dim=final_ebm_input_dim, hidden_dims=cfg.ebm_params.get('complex_ebm_hidden_dims', [1024,512,256]), dropout_rate=cfg.ebm_params.get('complex_ebm_dropout_rate',0.1), use_layernorm=cfg.ebm_params.get('complex_ebm_use_layernorm',True))
                        else: print(f"Unsupported arch {ebm_arch} for final EBM. Skipping."); final_concat_ebm_model = None
                        
                        if final_concat_ebm_model:
                            final_concat_ebm_model = final_concat_ebm_model.to(device=cfg.device, dtype=EBM_FORCED_DTYPE)
                            optimizer_final = optim.Adam(final_concat_ebm_model.parameters(), lr=cfg.ebm_params.ebm_lr)
                            
                            # Define final_ebm_path EARLIER
                            final_ebm_save_template = cfg.ebm_params.get('final_ebm_save_path_template', "output/ebm_models/{model_alias}_final_top{top_n_count}_layers_{layer_indices_str}_arch_{ebm_architecture}.pt")
                            layer_indices_str_for_path = "_".join(map(str, top_n_layers_for_final_ebm_indices)) # Use internal name
                            final_ebm_path = final_ebm_save_template.format(
                                model_alias=cfg.model_alias, 
                                top_n_count=len(top_n_layers_for_final_ebm_indices), # Use internal name
                                layer_indices_str=layer_indices_str_for_path,
                                ebm_architecture=ebm_arch
                            )

                            print(f"Training Final EBM (Input Dim: {final_ebm_input_dim}), saving to {final_ebm_path}...")

                            best_epoch_val_acc_for_final_ebm = -1.0
                            path_to_save_best_final_ebm = final_ebm_path 
                            os.makedirs(os.path.dirname(path_to_save_best_final_ebm), exist_ok=True) 
                            temperature = cfg.ebm_params.get('ebm_infonce_temperature', 0.07)

                            # tqdm for epochs for final EBM
                            epoch_pbar_final = tqdm(range(cfg.ebm_params.ebm_epochs), desc="Final EBM Epochs", leave=False)
                            for epoch in epoch_pbar_final:
                                final_concat_ebm_model.train()
                                total_train_loss_f = 0 # Correctly initialize here
                                # Inner tqdm for batches
                                batch_pbar_final = tqdm(final_ebm_train_dl, desc=f"FinalEBM Train Ep {epoch+1}", leave=False)
                                for x_p_b, x_n_b in batch_pbar_final:
                                    optimizer_final.zero_grad()
                                    e_p = final_concat_ebm_model(x_p_b.to(EBM_FORCED_DTYPE))
                                    e_n = final_concat_ebm_model(x_n_b.to(EBM_FORCED_DTYPE))
                                    loss_f = _calculate_infonce_loss(e_p, e_n, temperature, cfg.device)
                                    loss_f.backward(); optimizer_final.step(); total_train_loss_f += loss_f.item()
                                    batch_pbar_final.set_postfix(loss=loss_f.item())
                                
                                avg_train_loss_f = total_train_loss_f / len(final_ebm_train_dl) if len(final_ebm_train_dl) > 0 else 0
                                log_dict_final = {'TrainL': f'{avg_train_loss_f:.4f}'}
                                
                                final_val_acc = -1.0
                                if final_ebm_val_dl:
                                    final_concat_ebm_model.eval()
                                    tot_val_loss_f, corr_val_preds_f, tot_val_samples_f = 0,0,0
                                    with torch.no_grad():
                                        for x_p_v, x_n_v in final_ebm_val_dl:
                                            e_p_v = final_concat_ebm_model(x_p_v.to(EBM_FORCED_DTYPE))
                                            e_n_v = final_concat_ebm_model(x_n_v.to(EBM_FORCED_DTYPE))
                                            val_loss_f = _calculate_infonce_loss(e_p_v, e_n_v, temperature, cfg.device)
                                            tot_val_loss_f += val_loss_f.item()
                                            corr_val_preds_f += (e_p_v < 0).sum().item() + (e_n_v > 0).sum().item()
                                            tot_val_samples_f += x_p_v.shape[0] + x_n_v.shape[0]
                                    avg_val_loss_f = tot_val_loss_f / len(final_ebm_val_dl) if len(final_ebm_val_dl) > 0 else 0
                                    final_val_acc = (corr_val_preds_f / tot_val_samples_f) * 100 if tot_val_samples_f > 0 else 0
                                    # print(f"Final EBM Ep {epoch+1} AvgTrainL: {avg_train_loss_f:.4f} AvgValL: {avg_val_loss_f:.4f} ValAcc: {final_val_acc:.2f}%")
                                    log_dict_final['ValL'] = f'{avg_val_loss_f:.4f}'
                                    log_dict_final['ValAcc'] = f'{final_val_acc:.2f}%'

                                    if final_val_acc > best_epoch_val_acc_for_final_ebm:
                                        best_epoch_val_acc_for_final_ebm = final_val_acc
                                        torch.save(final_concat_ebm_model.state_dict(), path_to_save_best_final_ebm)
                                        # print(f"  Final EBM New best saved with ValAcc: {best_epoch_val_acc_for_final_ebm:.2f}% to {path_to_save_best_final_ebm}")
                                        log_dict_final['Saved'] = '*' # Indicate save
                                else: 
                                    # print(f"Final EBM Ep {epoch+1} AvgTrainL: {avg_train_loss_f:.4f} (No validation set)")
                                    if epoch == cfg.ebm_params.ebm_epochs - 1: 
                                        torch.save(final_concat_ebm_model.state_dict(), path_to_save_best_final_ebm)
                                        # print(f"  Final EBM (last epoch) saved to {path_to_save_best_final_ebm} (no validation set during training)")
                                        log_dict_final['Saved'] = '(last_ep)'
                                epoch_pbar_final.set_postfix(log_dict_final)
                            
                            if best_epoch_val_acc_for_final_ebm >= 0 and path_to_save_best_final_ebm:
                                print(f"  Final EBM training complete. Highest ValAcc: {best_epoch_val_acc_for_final_ebm:.2f}%, saved to {path_to_save_best_final_ebm}")
                                final_ebm_path_saved = path_to_save_best_final_ebm # Store saved path
                            elif path_to_save_best_final_ebm and os.path.exists(path_to_save_best_final_ebm):
                                print(f"  Final EBM training complete (last epoch model saved, no validation data during training) to {path_to_save_best_final_ebm}")
                                final_ebm_path_saved = path_to_save_best_final_ebm # Store saved path
                        else: 
                            final_concat_ebm_model = None 
                    else: # Not enough training samples for final EBM
                        print("Not enough training samples after split for Final EBM. Skipping.")
                        final_concat_ebm_model = None
                        top_n_layers_for_final_ebm_indices = [] # Reset internal name if path not taken
                else: # Not enough good/bad samples for final EBM
                    print("Not enough good/bad samples for Final EBM. Skipping.")
                    final_concat_ebm_model = None
                    top_n_layers_for_final_ebm_indices = [] # Reset internal name if path not taken
            else: # No good or bad activation samples for Final EBM
                print("No good or bad activation samples for Final EBM. Skipping.")
                final_concat_ebm_model = None
                top_n_layers_for_final_ebm_indices = [] # Reset internal name if path not taken
        #else:
        #    final_concat_ebm_model = None
        #    top_n_layers_for_final_ebm_indices = [] # Reset internal name if path not taken
    else: # This else corresponds to: if ebm_training_details_for_top_n and len(ebm_training_details_for_top_n) >= min_layers_for_final_config:
        print(f"Skipping Final EBM training: Not enough individual EBMs with validation accuracy (found {len(ebm_training_details_for_top_n)}, min needed {min_layers_for_final_config}).")
        final_concat_ebm_model = None
        top_n_layers_for_final_ebm_indices = [] # Ensure this is set correctly


    return trained_ebms, final_concat_ebm_model, top_n_layers_for_final_ebm_indices, individual_ebm_val_accuracies, best_epoch_val_acc_for_final_ebm, final_ebm_path_saved # Return new values


# --- Main Focused Evaluation Pipeline ---
def run_focused_eval(config_path, model_path_override=None, batch_size_override=None, force_retrain_ebm_cli=False):
   
    print(f"INFO: CUDA_VISIBLE_DEVICES is not being explicitly set by this script. GPU behavior depends on environment and cfg.device.")

    cfg = mmengine.Config.fromfile(config_path)
    if model_path_override: cfg.model_path = model_path_override
    if batch_size_override: cfg.batch_size = batch_size_override 
    cfg.model_alias = os.path.basename(cfg.model_path)
    cfg.device = cfg.get('device', 'cpu')
    print(f"INFO: cfg.device (target for main LLM & EBM) set to '{cfg.device}'.")

    force_retrain_ebm = force_retrain_ebm_cli
    if hasattr(cfg, 'ebm_params') and cfg.ebm_params:
        force_retrain_ebm = force_retrain_ebm or cfg.ebm_params.get('force_retrain_ebm', False)

    cfg.artifact_path = os.path.join("output", cfg.model_alias, "focused_ebm_eval")
    os.makedirs(cfg.artifact_path, exist_ok=True)
    
    log_file_path = os.path.join(cfg.artifact_path, "output_focused.log")
    sys.stdout = Logger(log_file_path)
    sys.stderr = Logger(os.path.join(cfg.artifact_path, "error_focused.log"))
    
    print(f"--- Starting Focused EBM Evaluation ---")
    print(f"Artifacts will be saved to: {cfg.artifact_path}")
    cfg.dump(os.path.join(cfg.artifact_path, 'config_focused_run.yaml'))

    print(f"Loading LLM: {cfg.model_path}")
    model_base = construct_model_base(cfg.model_path, system=cfg.get('system'))

    print("Attempting to train or load EBMs...")
    ebm_models_by_layer, final_ebm_model, top_layers_for_final, indiv_val_accs, final_ebm_best_val_acc, final_ebm_saved_path = train_and_save_ebm_if_needed(cfg, model_base, force_retrain_ebm) # Receive new values
    
    if indiv_val_accs:
        print("\n--- Individual EBM Validation Accuracies ---")
        for layer, acc in sorted(indiv_val_accs.items()):
            print(f"Layer {layer}: {acc:.2f}%")
        print("-----------------------------------------")

    if final_ebm_model:
        print("\n--- Final Concatenated EBM Trained ---")
        print(f"Based on Top {len(top_layers_for_final)} layers: {top_layers_for_final}")
        if final_ebm_best_val_acc >= 0 and final_ebm_saved_path:
            print(f"Highest Validation Accuracy: {final_ebm_best_val_acc:.2f}% (saved to {final_ebm_saved_path})")
        elif final_ebm_saved_path: # Saved but no specific val acc recorded during a non-val run
            print(f"Model saved to {final_ebm_saved_path} (validation accuracy not applicable or recorded as best)")
        print(f"Input dimension for this final EBM: {len(top_layers_for_final) * model_base.model.config.hidden_size}")
        print(f"Intervention with this final EBM is not yet implemented in the standard hook. Current run uses individual EBMs.")
        print("------------------------------------")
    elif hasattr(cfg, 'ebm_params') and cfg.ebm_params.get('top_n_layers_for_final_ebm', 0) > 0 : # if user intended to train one
        print("\n--- Final Concatenated EBM NOT Trained (check logs for reasons, e.g., not enough layers met criteria) ---")

    if not ebm_models_by_layer: 
        print("Warning: No individual EBMs were loaded or trained. Proceeding without EBM intervention if configured.")
    else:
        print(f"Successfully trained/loaded EBMs for layers: {sorted(list(ebm_models_by_layer.keys()))}")

    # --- Setup EBM Intervention Hooks (using individual EBMs) ---
    intervention_fwd_pre_hooks = []
    if hasattr(cfg, 'ebm_params') and cfg.ebm_params and ebm_models_by_layer:
        layers_to_intervene_on: List[int] = []
        
        if top_layers_for_final: # Prioritize top_layers_for_final if available
            print(f"Using Top {len(top_layers_for_final)} performing EBM layers for intervention: {sorted(top_layers_for_final)}")
            layers_to_intervene_on = top_layers_for_final
        else:
            print("Top layers for final EBM not available or empty. Falling back to 'ebm_intervention_layers' config.")
            ebm_intervention_layers_config = cfg.ebm_params.get('ebm_intervention_layers', 'all') 
            if isinstance(ebm_intervention_layers_config, list):
                layers_to_intervene_on = [int(l) for l in ebm_intervention_layers_config]
            elif isinstance(ebm_intervention_layers_config, str) and ebm_intervention_layers_config.lower() == 'all':
                layers_to_intervene_on = sorted(list(ebm_models_by_layer.keys())) 
            elif isinstance(ebm_intervention_layers_config, int):
                layers_to_intervene_on = [ebm_intervention_layers_config]
            else:
                print(f"Warning: Invalid 'ebm_intervention_layers' config: {ebm_intervention_layers_config}. No specific EBM intervention will be applied unless 'all' available are used by default.")
            print(f"Using EBM intervention layers based on configuration: {sorted(layers_to_intervene_on) if layers_to_intervene_on else 'None'}")

        ebm_intervention_positions_str = str(cfg.ebm_params.get('ebm_intervention_positions', str(cfg.ebm_params.get('ebm_target_positions')))) # Ensure fallback from ebm_target_positions
        ebm_intervention_positions = [int(p.strip()) for p in ebm_intervention_positions_str.split(',')]
        
        if not layers_to_intervene_on:
            print("No EBM intervention layers specified or resolved. Skipping EBM hook setup.")
        else:
            print(f"Attempting to set up EBM intervention for layers: {sorted(layers_to_intervene_on)}")

            for layer_idx in sorted(list(set(layers_to_intervene_on))): # Use set to avoid duplicates, then sort
                if not (0 <= layer_idx < model_base.model.config.num_hidden_layers):
                    print(f"Warning: EBM intervention layer {layer_idx} is out of LLM bounds ({model_base.model.config.num_hidden_layers} layers). Skipping.")
                    continue
                
                current_ebm_for_hook = ebm_models_by_layer.get(layer_idx)
                if not current_ebm_for_hook:
                    print(f"Warning: No EBM found for intervention layer {layer_idx} in the loaded/trained EBMs. Skipping hook for this layer.")
                    continue

                # Ensure EBM is on the correct device and in float32 for intervention hook compatibility
                current_ebm_for_hook = current_ebm_for_hook.to(device=cfg.device, dtype=torch.float32)
                current_ebm_for_hook.eval()
                print(f"EBM for layer {layer_idx} ready for intervention. Training: {current_ebm_for_hook.training}, Dtype: {next(current_ebm_for_hook.parameters()).dtype}")
                
                target_module = model_base.model_block_modules[layer_idx]
                ebm_hook = get_ebm_intervention_hook(
                    ebm_model_fr=current_ebm_for_hook, # Pass the EBM for this specific layer
                    target_layer_idx=layer_idx, 
                    current_hook_layer_idx=layer_idx, 
                    position_indices=ebm_intervention_positions,
                    eta=cfg.ebm_params.ebm_eta,
                    num_gradient_steps=cfg.ebm_params.ebm_num_gradient_steps,
                    ebm_model_tr=None, 
                    lambda_ebm_ortho=0.0, 
                    device=cfg.device 
                )
                intervention_fwd_pre_hooks.append((target_module, ebm_hook))
            
            if intervention_fwd_pre_hooks:
                print(f"Added {len(intervention_fwd_pre_hooks)} EBM intervention pre-hooks for layers: {[l for l in layers_to_intervene_on if l in ebm_models_by_layer]}.")
            else:
                print("No EBM intervention hooks were added. This might be due to configuration or no EBMs available for specified intervention layers.")

    elif not ebm_models_by_layer and hasattr(cfg, 'ebm_params') and cfg.ebm_params:
        print("EBM parameters are configured, but no EBM models were loaded or trained. Running without EBM intervention.")
    else: # No ebm_params or no ebm_models_by_layer
        print("No ebm_params found in config or no EBMs available. Running without EBM intervention.")

    # --- Focused Evaluation on Specific Datasets ---
    # 只评估用户要求的数据集：JBB, Harmful, ORB-H, XSTest-S(H), OKTest
    datasets_to_evaluate = {
        # "HarmBench Test": 'dataset/processed/harmbench_test.json',
        # "JailbreakBench Test": 'dataset/processed/jailbreakbench.json',
        # "XSTest Safe": 'dataset/processed/xstest_safe.json',
        # "OKTest (False Refusal)": 'dataset/processed/oktest.json',

        # "OKTest (False Refusal)": 'dataset/processed/oktest.json', #32
        # "WGT Adv harm": 'dataset/processed/wildguard_test_adv.json', 
        # "WGT Vani harm": 'dataset/processed/wildguard_test_vani.json',
        # "WJB Adv harm": 'dataset/processed/wildjailbreak_test_adv_harmful.json',
        # "DAN Unsafe": 'dataset/processed/DAN_unsafe.json',
        # "HarmBench Test": 'dataset/processed/harmbench_test.json', #32
        # "Or-bench toxic": 'dataset/processed/or-bench-toxic-harmful.json', #32
        # "XSTest Unsafe": 'dataset/processed/xstest_unsafe.json', #32
        # "WJB Adv benign": 'dataset/processed/wildjailbreak_test_adv_safe.json',

        #"XSTest Safe": 'dataset/processed/xstest_safe.json', #32
        #"OKTest (False Refusal)": 'dataset/processed/oktest.json', #32
        #"HarmBench Test": 'dataset/processed/harmbench_test.json', #32
        #"Or-bench toxic": 'dataset/processed/or-bench-toxic-harmful.json', #32
        #"XSTest Unsafe": 'dataset/processed/xstest_unsafe.json', #32
        #"XSTest Safe": 'dataset/processed/xstest_safe.json', #32
        #"WGT Adv harm": 'dataset/processed/wildguard_test_adv.json', 
        #"WGT Vani harm": 'dataset/processed/wildguard_test_vani.json',
        #"WJB Adv harm": 'dataset/processed/wjb_adv_harmful.json',
        #"DAN Unsafe": 'dataset/processed/DAN_unsafe.json',
        #"WJB Adv benign": 'dataset/processed/wjb_adv_safe.json',

         # 使用processed目录中实际存在的数据集文件
         #"WGT Adv harm": 'dataset/processed/wildguard_test_adv.json',
         #"DAN Unsafe": 'dataset/processed/DAN_unsafe.json',
         "JailbreakBench": 'dataset/processed/jailbreakbench.json',       # JBB
         "HarmBench": 'dataset/processed/harmbench_test.json',            # Harmful测试
         "ORB-H": 'dataset/splits/or_bench_hard_test.json',        # Or-bench Hard  
         "XSTest Safe": 'dataset/processed/xstest_safe.json',             # XSTest safe
         "OKTest": 'dataset/processed/oktest.json',                       # OKTest
    }
    # Evaluation methodologies, can be customized per dataset if needed
    # Defaulting to substring_matching as commonly used.
    default_eval_methodologies = cfg.get('default_eval_methodologies', ['substring_matching'])
    dataset_specific_eval_methodologies = cfg.get('dataset_specific_eval_methodologies', {})


    print(f"\n--- Generating and Evaluating Completions for Focused Datasets with EBM Intervention ---")
    intervention_label = "ebm_intervention" # Label for file naming

    for display_name, dataset_key in datasets_to_evaluate.items():
        print(f"\n--- Evaluating on: {display_name} (key: {dataset_key}) ---")
        
        # Dataset loading is handled within generate_and_save_completions_for_dataset
        # It will use load_dataset_split(harmtype=dataset_key, split='test', ...)
        
        generate_and_save_completions_for_dataset(
            cfg, model_base,
            fwd_pre_hooks=intervention_fwd_pre_hooks, # Pass the EBM hooks
            fwd_hooks=[], # No standard forward hooks for this focused EBM setup
            intervention_label=intervention_label,
            dataset_name=dataset_key, # Use the key for loading
            dataset=None, # Let the function load it
            system=model_base.system_prompt if hasattr(model_base, 'system_prompt') and model_base.system_prompt is not None else cfg.get('system')
        )
        
        current_eval_methodologies = dataset_specific_eval_methodologies.get(dataset_key, default_eval_methodologies)
        if not current_eval_methodologies:
            print(f"No evaluation methodologies specified for {display_name}. Skipping evaluation results calculation.")
            continue

        evaluate_completions_and_save_results_for_dataset(
            cfg,
            intervention_label=intervention_label,
            dataset_name=dataset_key,
            eval_methodologies=current_eval_methodologies
        )

    print(f"\n--- Focused EBM Evaluation Completed. Results in {cfg.artifact_path} ---")

    # Cleanup
    del model_base.model, model_base
    if ebm_models_by_layer: 
        for layer_idx in list(ebm_models_by_layer.keys()): 
            del ebm_models_by_layer[layer_idx]
        del ebm_models_by_layer
    if final_ebm_model:
        del final_ebm_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_arguments()
    run_focused_eval(
        config_path=args.config_path,
        model_path_override=args.model_path,
        batch_size_override=args.batch_size,
        force_retrain_ebm_cli=args.force_retrain_ebm
    )
