import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from einops import rearrange

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook

def get_raw_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules, batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # Pre-allocate the cache to store activations for all samples
    raw_activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    for i in tqdm(range(0, n_samples, batch_size)):
        # Get the batch indices
        batch_indices = list(range(i, min(i + batch_size, n_samples)))
        batch_instructions = instructions[i:i + batch_size]
        inputs = tokenize_instructions_fn(instructions=batch_instructions)

        # Define the hooks for the current batch
        fwd_pre_hooks = [
            (
                block_modules[layer],
                get_raw_activations_pre_hook(
                    layer=layer,
                    cache=raw_activations,
                    batch_indices=batch_indices,
                    positions=positions
                ),
            )
            for layer in range(n_layers)
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return raw_activations

def get_raw_activations_pre_hook(layer, cache, batch_indices, positions):
    def hook_fn(module, input):
        # Extract the activations from the input
        activation = input[0]  # Shape: (batch_size, seq_len, d_model)
        # Select the specified positions
        activation_selected = activation[:, positions, :]  # Shape: (batch_size, len(positions), d_model)
        # Compute the mean over the selected positions
        activation_mean = activation_selected.mean(dim=1)  # Shape: (batch_size, d_model)
        # Update the cache at the correct indices
        cache[batch_indices, layer, :] = activation_mean.to(cache.dtype)
    return hook_fn

def refusal_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    refusal_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model, instructions, tokenize_instructions_fn, refusal_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32, system=None):
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)

    refusal_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size], system=system)

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        refusal_scores[i:i+batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores

def get_last_position_logits(model, tokenizer, instructions, tokenize_instructions_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32) -> Float[Tensor, "n_instructions d_vocab"]:
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits

def plot_refusal_scores(
    refusal_scores: Float[Tensor, 'n_pos n_layer'],
    baseline_refusal_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
    y_label: str = 'Refusal score',
    start_layer: int = 0,
):
    n_pos, n_layer = refusal_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            [x + start_layer for x in list(range(n_layer))],
            refusal_scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_refusal_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_refusal_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_refusal_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel(y_label)
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

# returns True if the direction should be filtered out
def filter_fn_ablation(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    # if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
    #     return True
    return False

def filter_fn_addition(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True

def select_direction(
    cfg,
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    pair_name: str,
    artifact_dir,
    kl_threshold= 0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=32,
    mode = 'ablation',
    top_n = 1
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    baseline_refusal_scores_harmful = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)

    # save the baseline refusal scores
    with open(f"{artifact_dir}/baseline_refusal_scores_harmful_{mode}.json", 'w') as f:
        json.dump(baseline_refusal_scores_harmful.mean().item(), f, indent=4)
    
    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    addition_kl_div_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_harmless_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores_harmful_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    
    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()


    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):
            
            ablation_dir = candidate_directions[source_pos, source_layer]
            
            coeff = torch.tensor(cfg.addact_coeff)
            fwd_pre_hooks = [(model_base.model_block_modules[source_layer+cfg.start_layer], get_activation_addition_input_pre_hook(vector=ablation_dir, coeff=coeff))]
            fwd_hooks = []
            
            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            addition_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    
    
    
    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]

            refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):

            refusal_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(cfg.addact_coeff)

            layer_to_add = source_layer + cfg.start_layer
            fwd_pre_hooks = [(model_base.model_block_modules[layer_to_add], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)[cfg.start_layer:]]

            refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_harmless_scores[source_pos, source_layer] = refusal_scores.mean().item()
    
    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):

            refusal_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(cfg.addact_coeff)
            layer_to_add = source_layer + cfg.start_layer
            fwd_pre_hooks = [(model_base.model_block_modules[layer_to_add], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores_harmful_scores[source_pos, source_layer] = refusal_scores.mean().item()

    
    
    
    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'Ablating direction on {pair_name[0]} instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'ablation_scores_{pair_name[0]}_{pair_name[1]}',
        start_layer=cfg.start_layer
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'Adding direction on {pair_name[1]} instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'actadd_scores_{pair_name[0]}_{pair_name[1]}',
        start_layer=cfg.start_layer
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'KL Divergence when ablating direction on {pair_name[1]} instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'ablation_kl_div_scores_{pair_name[0]}_{pair_name[1]}',
        y_label='KL Divergence',
        start_layer=cfg.start_layer
    )
    
    plot_refusal_scores(
        refusal_scores=addition_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'KL Divergence when adding direction on {pair_name[1]} instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'actadd_kl_div_scores_{pair_name[0]}_{pair_name[1]}',
        y_label='KL Divergence',
        start_layer=cfg.start_layer
    )

    plot_refusal_scores(
        refusal_scores=ablation_refusal_harmless_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'Ablating direction on {pair_name[1]} instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'ablation_scores_{pair_name[0]}_{pair_name[1]}_on_{pair_name[1]}',
        start_layer=cfg.start_layer
    )
    
    plot_refusal_scores(
        refusal_scores=steering_refusal_scores_harmful_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'Adding direction on {pair_name[0]} instructions',
        artifact_dir=artifact_dir,
        artifact_name=f'actadd_scores_over_refusal_{pair_name[0]}_{pair_name[1]}_on_{pair_name[0]}',
        start_layer=cfg.start_layer
    )
    
    
    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer + cfg.start_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            if mode == 'ablation':
            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
                sorting_score = -refusal_score
            # we filter out directions if the KL threshold 
                kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()
                discard_direction = filter_fn_ablation(
                    refusal_score=refusal_score,
                    steering_score=steering_score,
                    kl_div_score=kl_div_score,
                    layer=source_layer + cfg.start_layer,
                    n_layer=n_layer + cfg.start_layer,
                    kl_threshold=kl_threshold,
                    induce_refusal_threshold=induce_refusal_threshold,
                    prune_layer_percentage=prune_layer_percentage
                )
            elif mode == 'addition':
                sorting_score = steering_score
                kl_div_score = addition_kl_div_scores[source_pos, source_layer].item()
                discard_direction = filter_fn_addition(
                    refusal_score=refusal_score,
                    steering_score=steering_score,
                    kl_div_score=kl_div_score,
                    layer=source_layer + cfg.start_layer,
                    n_layer=n_layer + cfg.start_layer,
                    kl_threshold=kl_threshold,
                    induce_refusal_threshold=induce_refusal_threshold,
                    prune_layer_percentage=prune_layer_percentage
                )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer + cfg.start_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': kl_div_score
            })   

    with open(f"{artifact_dir}/direction_evaluations_{mode}.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['refusal_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered_{mode}.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # if top_n == 1:
        
    #     # now return the best position, layer, and direction
    #     score, pos, layer = filtered_scores[0]
    # else: 
        # now return the best n positions, layers, and directions
    top_n_scores = filtered_scores[:top_n]

# Initialize lists to collect results
    positions = []
    layers = []
    directions = []

    # Loop through the top n scores and print the relevant information
    for score, pos, layer in top_n_scores:
        print(f"Selected direction: position={pos}, layer={layer + cfg.start_layer}")
        print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
        print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
        print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
        
        # Append results to lists
        positions.append(pos)
        layers.append(layer + cfg.start_layer)
        directions.append(candidate_directions[pos, layer])

    # Return the collected results
    return positions, layers, directions


def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)