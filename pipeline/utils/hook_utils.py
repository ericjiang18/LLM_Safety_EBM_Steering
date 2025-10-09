import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Tuple, Callable, Any, Optional
from jaxtyping import Float
# --- Context Manager for Hooks ---
class add_hooks(torch.nn.Module):
    """
    Context manager for adding and removing hooks.
    Slightly adapted from common implementations.
    """
    def __init__(self, module_forward_pre_hooks: List[Tuple[nn.Module, Callable]] = None,
                 module_forward_hooks: List[Tuple[nn.Module, Callable]] = None):
        super().__init__()
        self.module_forward_pre_hooks = module_forward_pre_hooks if module_forward_pre_hooks else []
        self.module_forward_hooks = module_forward_hooks if module_forward_hooks else []
        self.handles = []

    def __enter__(self):
        for module, hook_fn in self.module_forward_pre_hooks:
            if hook_fn is None: # Added safety check
                print(f"Warning: Attempted to register a None pre-hook for module {module}")
                continue
            self.handles.append(module.register_forward_pre_hook(hook_fn))
        for module, hook_fn in self.module_forward_hooks:
            if hook_fn is None: # Added safety check
                print(f"Warning: Attempted to register a None forward hook for module {module}")
                continue
            self.handles.append(module.register_forward_hook(hook_fn))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

# --- Existing Vector-Based Intervention Hooks (from paper's methodology) ---

def get_activation_addition_input_pre_hook(
    vector: Float[Tensor, "d_model"],
    coeff: float = 1.0
) -> Callable:
    """
    Returns a forward pre-hook that adds a scaled vector to the input activation.
    Typically applied to the input of a model block (e.g., a Transformer layer).
    """
    if isinstance(coeff, torch.Tensor): # Ensure coeff is a float or compatible
        coeff_val = coeff.item()
    else:
        coeff_val = coeff

    def hook_fn(module: nn.Module, input_tensors: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        # input_tensors[0] is usually the hidden_states: (batch_size, seq_len, hidden_dim)
        original_activations = input_tensors[0]
        # Vector needs to be on the same device and dtype as activations
        vec_to_add = vector.to(original_activations.device, dtype=original_activations.dtype)
        
        # Add to all token positions in the sequence for the specified layer input
        modified_activations = original_activations + coeff_val * vec_to_add.unsqueeze(0).unsqueeze(0) 
        return (modified_activations,) + input_tensors[1:]
    return hook_fn

def _project_and_subtract(
    activation_tensor: Float[Tensor, "... d_model"],
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0
) -> Float[Tensor, "... d_model"]:
    """Helper to project activation onto direction and subtract scaled projection."""
    direction_norm_sq = torch.sum(direction * direction)
    if direction_norm_sq < 1e-8: # Avoid division by zero if direction is near zero
        return activation_tensor
    
    proj_coeffs = torch.einsum("...d,d->...", activation_tensor, direction) / direction_norm_sq
    projection = proj_coeffs.unsqueeze(-1) * direction.unsqueeze(0) # Adjust for broadcasting
    return activation_tensor - coeff * projection

def get_direction_ablation_input_pre_hook(
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0
) -> Callable:
    """
    Returns a forward pre-hook that ablates a direction from the input activation of a module.
    Typically applied to the input of a model block (e.g., model.model.layers[i]).
    """
    def hook_fn(module: nn.Module, input_tensors: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        original_activations = input_tensors[0] # (batch, seq, hidden_dim)
        vec_to_ablate = direction.to(original_activations.device, dtype=original_activations.dtype)
        
        modified_activations = _project_and_subtract(original_activations, vec_to_ablate, coeff)
        return (modified_activations,) + input_tensors[1:]
    return hook_fn

def get_direction_ablation_output_hook(
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0
) -> Callable:
    """
    Returns a forward hook that ablates a direction from the *output* activation of a module.
    Typically applied to the output of an attention or MLP sub-block.
    """
    def hook_fn(module: nn.Module, input_tensors: Tuple[torch.Tensor], output_tensors: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        # For many nn.Modules, output_tensors is a tuple where the first element is the main activation
        original_output_activations = output_tensors[0] # (batch, seq, hidden_dim)
        vec_to_ablate = direction.to(original_output_activations.device, dtype=original_output_activations.dtype)

        modified_output_activations = _project_and_subtract(original_output_activations, vec_to_ablate, coeff)
        return (modified_output_activations,) + output_tensors[1:]
    return hook_fn

def get_all_direction_ablation_hooks(
    model_base, # Instance of ModelBase
    direction: Float[Tensor, "d_model"],
    start_layer: int,
    coeff: float = 1.0
) -> Tuple[List[Tuple[nn.Module, Callable]], List[Tuple[nn.Module, Callable]]]:
    """
    Creates ablation hooks for all relevant layers (blocks, attention outputs, MLP outputs).
    The paper mentions "This operation is done across all layers and token positions..." for removal.
    It also mentions (Appendix D) equivalence to model editing on Att and FFN blocks.
    This implementation will add hooks to ablate from the *output* of Attn and MLP,
    and *input* of the next block, which is a common way to achieve ablation from residual stream.
    """
    fwd_pre_hooks = []
    fwd_hooks = [] # For output hooks

    # Ablate from the input of each subsequent block (resid_pre)
    for layer_idx in range(start_layer, model_base.model.config.num_hidden_layers):
        # Input to the block (after previous layer's Attn/MLP and LayerNorm)
        block_module = model_base.model_block_modules[layer_idx]
        fwd_pre_hooks.append(
            (block_module, get_direction_ablation_input_pre_hook(direction=direction, coeff=coeff))
        )
        
        # Alternatively, or additionally, ablate from outputs of Attn and MLP within the layer
        # if model_base.model_attn_modules and layer_idx < len(model_base.model_attn_modules):
        #     attn_module = model_base.model_attn_modules[layer_idx]
        #     fwd_hooks.append( # Output hook for attention
        #         (attn_module, get_direction_ablation_output_hook(direction=direction, coeff=coeff))
        #     )
        # if model_base.model_mlp_modules and layer_idx < len(model_base.model_mlp_modules):
        #     mlp_module = model_base.model_mlp_modules[layer_idx]
        #     fwd_hooks.append( # Output hook for MLP
        #         (mlp_module, get_direction_ablation_output_hook(direction=direction, coeff=coeff))
        #     )
            
    # The paper's Figure 7 shows ablation on "Residual Stream" which could mean
    # input to each block, or output of each sub-block before adding back.
    # The `run_pipeline.py` code seems to add pre_hooks for blocks and forward_hooks for attn/mlp.
    # The provided `select_direction.py` uses `get_direction_ablation_input_pre_hook` for block_modules
    # and `get_direction_ablation_output_hook` for attn_modules and mlp_modules.
    # So, `get_all_direction_ablation_hooks` likely aims to replicate this broad application.

    # Let's follow the structure implied by run_pipeline.py more closely:
    # `pre_hooks` from get_all_direction_ablation_hooks become part of `harm_ablation_fwd_pre_hooks`
    # `fwd_hooks` from get_all_direction_ablation_hooks become part of `harm_ablation_fwd_hooks`
    
    # Option 1: Ablate input to each block (as done above, this list is for fwd_pre_hooks)
    
    # Option 2: Ablate outputs of Attn and MLP (this list is for fwd_hooks)
    # fwd_hooks_output = []
    # for layer_idx in range(start_layer, model_base.model.config.num_hidden_layers):
    #     if hasattr(model_base, 'model_attn_modules') and layer_idx < len(model_base.model_attn_modules):
    #          fwd_hooks_output.append((model_base.model_attn_modules[layer_idx], get_direction_ablation_output_hook(direction=direction, coeff=coeff)))
    #     if hasattr(model_base, 'model_mlp_modules') and layer_idx < len(model_base.model_mlp_modules):
    #          fwd_hooks_output.append((model_base.model_mlp_modules[layer_idx], get_direction_ablation_output_hook(direction=direction, coeff=coeff)))
    # return fwd_pre_hooks, fwd_hooks_output

    # To match `run_pipeline.py`'s structure, it implies that the `get_all_direction_ablation_hooks`
    # should return both pre-hooks (for block inputs) and regular forward hooks (for Attn/MLP outputs).
    # The example from `select_direction.py` applies it to all layers *starting from cfg.start_layer*.
    # The paper says "across all layers". Let's make this configurable or clear.
    # For simplicity and alignment with `select_direction.py`'s detailed usage:
    
    pre_hooks_for_blocks = []
    # forward_hooks_for_submodules = [] # This was commented out, let's keep it that way unless explicitly needed

    for layer_idx in range(start_layer, model_base.model.config.num_hidden_layers):
        # Pre-hook for the input of the entire block
        block_module = model_base.model_block_modules[layer_idx]
        pre_hooks_for_blocks.append(
            (block_module, get_direction_ablation_input_pre_hook(direction=direction, coeff=coeff))
        )
        # Forward (output) hooks for attention and MLP sub-modules
        # This might double-ablate if input to block is already ablated and sub-modules also ablate.
        # The original `select_direction.py` in the prompt implied these are separate strategies.
        # The `run_pipeline.py` in the prompt combines `pre_hooks` and `fwd_hooks` from this function.
        # Let's assume for now it means ablating *outputs* of Attn and MLP.
        # if model_base.model_attn_modules and layer_idx < len(model_base.model_attn_modules):
        #     attn_module = model_base.model_attn_modules[layer_idx]
        #     forward_hooks_for_submodules.append(
        #         (attn_module, get_direction_ablation_output_hook(direction=direction, coeff=coeff))
        #     )
        # if model_base.model_mlp_modules and layer_idx < len(model_base.model_mlp_modules):
        #     mlp_module = model_base.model_mlp_modules[layer_idx]
        #     forward_hooks_for_submodules.append(
        #         (mlp_module, get_direction_ablation_output_hook(direction=direction, coeff=coeff))
        #     )
    
    # Given the user's `run_pipeline.py` snippet, `get_all_direction_ablation_hooks` returns two lists:
    # `pre_hooks` and `fwd_hooks`. `pre_hooks` are extended to `harm_ablation_fwd_pre_hooks` and
    # `fwd_hooks` are extended to `harm_ablation_fwd_hooks`.
    # This suggests the `pre_hooks` from this function are `forward_pre_hooks` and `fwd_hooks` are `forward_hooks`.
    # The `select_direction` snippet shows this function being called without a coeff, so default it to 1.0
    # It seems the intention of `get_all_direction_ablation_hooks` in `run_pipeline.py` is to gather
    # hooks that collectively achieve the ablation effect, potentially across different hook types.

    # Reconciling with the snippet from `run_pipeline.py`:
    # `pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_harm, cfg.start_layer, cfg.ablation_coeff)`
    # It's most likely that `pre_hooks` list contains tuples for `register_forward_pre_hook` and `fwd_hooks` for `register_forward_hook`.
    # The ablation can be applied at the input of a block (pre-hook) or at the output of components like Attn/MLP (forward hook).
    # Let's provide distinct options. The paper's Fig 7 shows ablation "on activation during inference" which is general.
    # Appendix D implies editing W_out of FFN/Attn. This means ablating the *output* of these blocks.
    
    # To align with how it seems to be used, let's make it return hooks for Attn and MLP outputs.
    # The `get_direction_ablation_input_pre_hook` might be used separately if needed.
    output_ablation_hooks = []
    # The previous version of this function returned pre_hooks_for_blocks and output_ablation_hooks
    # Let's stick to what was likely intended before: pre-hooks for blocks, and output hooks for submodules.
    # The `run_pipeline.py` example used `pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(...)`
    # So, the first list should be pre-hooks, the second should be forward_hooks.
    # The current `pre_hooks_for_blocks` is correct for the first element.
    # For the second, `output_ablation_hooks` (which are forward hooks for submodules) is appropriate.
    
    # Re-enabling output_ablation_hooks as per typical usage with this function name
    for layer_idx in range(start_layer, model_base.model.config.num_hidden_layers):
        if hasattr(model_base, 'model_attn_modules') and layer_idx < len(model_base.model_attn_modules):
             output_ablation_hooks.append(
                 (model_base.model_attn_modules[layer_idx], get_direction_ablation_output_hook(direction=direction, coeff=coeff))
            )
        if hasattr(model_base, 'model_mlp_modules') and layer_idx < len(model_base.model_mlp_modules):
             output_ablation_hooks.append(
                 (model_base.model_mlp_modules[layer_idx], get_direction_ablation_output_hook(direction=direction, coeff=coeff))
            )
    return pre_hooks_for_blocks, output_ablation_hooks

# --- New EBM Intervention Hook ---
def get_ebm_intervention_hook(
    ebm_model_fr: torch.nn.Module,
    target_layer_idx: int, # Informational: which EBM's "perspective" if multiple EBMs existed
    current_hook_layer_idx: int, # Actual layer this hook is attached to
    position_indices: List[int],
    eta: float,
    num_gradient_steps: int = 1,
    ebm_model_tr: Optional[torch.nn.Module] = None,
    lambda_ebm_ortho: float = 0.0,
    device: str = 'cuda'
) -> Callable:
    """
    Returns a forward pre-hook that modifies activations using EBM gradients.
    Assumes EBM models are already on the correct device (typically cfg.device).
    The EBM itself should be in float32 for stability, inputs will be cast.
    """
    # Ensure the EBM is in float32 and on the specified device.
    if ebm_model_fr:
        ebm_model_fr = ebm_model_fr.to(device=device, dtype=torch.float32) 
    if ebm_model_tr: # If a True Refusal EBM is used
        ebm_model_tr = ebm_model_tr.to(device=device, dtype=torch.float32)

    def hook_fn(module: torch.nn.Module, input_tensors: Tuple[torch.Tensor]) -> Optional[Tuple[torch.Tensor]]:
        
        # print(f"[EBM Hook Debug L{current_hook_layer_idx}] Called. Module: {type(module).__name__}, EBM_FR is None: {ebm_model_fr is None}")

        if not ebm_model_fr: # If no FR EBM, do nothing.
            # print(f"[EBM Hook L{current_hook_layer_idx}] ebm_model_fr is None. Skipping intervention.")
            return input_tensors

        # --- Input Activations Handling ---
        # input_tensors[0] is usually the hidden_states: (batch_size, seq_len, hidden_dim)
        original_activations_batch = input_tensors[0]
        
        # Determine actual indices for slicing based on position_indices (can be negative)
        seq_len = original_activations_batch.shape[1]
        actual_position_indices = []
        for pos_idx in position_indices:
            actual_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
            if 0 <= actual_idx < seq_len:
                actual_position_indices.append(actual_idx)
            else:
                # This case should ideally be handled by pre-filtering valid positions or warning.
                # For now, if an index is out of bounds, we might skip it or use a default.
                # Let's ensure only valid indices are used.
                # print(f"[EBM Hook Warning L{current_hook_layer_idx}] Position index {pos_idx} (actual: {actual_idx}) is out of bounds for seq_len {seq_len}. Skipping this position.")
                pass # Silently skip invalid positions for now
        
        if not actual_position_indices: # If no valid positions, return original
            # print(f"[EBM Hook Info L{current_hook_layer_idx}] No valid positions for intervention. Positions: {position_indices}, SeqLen: {seq_len}. Skipping.")
            return input_tensors

        # Slice the activations at the target positions for intervention
        # act_slice_current_step will be (batch_size, num_target_positions, hidden_dim)
        try:
            act_slice_current_step = original_activations_batch[:, actual_position_indices, :].clone()
        except IndexError as e:
            # print(f"[EBM Hook Error L{current_hook_layer_idx}] IndexError during slicing. Positions: {actual_position_indices}, Shape: {original_activations_batch.shape}. Error: {e}")
            return input_tensors # Skip intervention if slicing fails

        if act_slice_current_step.numel() == 0:
            # print(f"[EBM Hook Info L{current_hook_layer_idx}] Activation slice is empty. Skipping intervention.")
            return input_tensors
            
        # Store original EBM training modes
        original_ebm_fr_training_mode = ebm_model_fr.training
        original_ebm_tr_training_mode = ebm_model_tr.training if ebm_model_tr else None

        # Ensure all EBM_FR parameters require grad for the intervention
        for name, param in ebm_model_fr.named_parameters():
            if not param.requires_grad:
                # print(f"  [EBM Hook Debug L{current_hook_layer_idx}] WARNING: EBM_FR param {name} was requires_grad=False. Forcing to True for intervention.")
                param.requires_grad_(True)

        modified_slice = act_slice_current_step.clone() # This is what we'll modify and put back

        # --- Iterative Gradient Ascent/Descent on the SLICE ---
        for step in range(num_gradient_steps): # Ensuring this is at 8 spaces
            # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] --- Starting Iteration ---")
            current_slice_for_grad = modified_slice.clone().detach().requires_grad_(True) # Ensuring this is at 12 spaces
            
            # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] Slice for grad: shape {current_slice_for_grad.shape}, requires_grad: {current_slice_for_grad.requires_grad}")

            with torch.enable_grad(): # Ensuring this is at 12 spaces
                # --- False Refusal (FR) EBM ---
                ebm_model_fr.train() # Set to train mode for gradient calculation # Ensuring this is at 16 spaces
                
                # Input to EBM should be (batch_size * num_target_positions, hidden_dim) if EBM expects 2D input
                # Or handle (batch, num_pos, hidden_dim) if EBM handles 3D (e.g. via reshaping internally or if it's designed for sequences)
                # Assuming EBM expects (N, hidden_dim)
                batch_size, num_pos, hidden_dim = current_slice_for_grad.shape
                input_to_ebm_fr_reshaped = current_slice_for_grad.reshape(-1, hidden_dim)
                
                # Cast to EBM's weight dtype (float32) and ensure it's on the hook's specified device
                input_to_ebm_fr_casted = input_to_ebm_fr_reshaped.to(device=device, dtype=torch.float32)
                
                # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] EBM_FR input (casted, reshaped): shape {input_to_ebm_fr_casted.shape}, dtype {input_to_ebm_fr_casted.dtype}, device {input_to_ebm_fr_casted.device}")

                energy_fr = ebm_model_fr(input_to_ebm_fr_casted) # Output shape: (batch_size * num_pos, 1) or (batch_size * num_pos)
                # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] EBM_FR energy: shape {energy_fr.shape}, mean {energy_fr.mean().item()}, requires_grad {energy_fr.requires_grad}, grad_fn {energy_fr.grad_fn}")

                if not original_ebm_fr_training_mode: ebm_model_fr.eval() # Restore EBM to eval if it was originally

                if energy_fr.grad_fn is None:
                    # print(f"WARNING L{current_hook_layer_idx} Step {step+1}: energy_fr has no grad_fn! Skipping grad step for FR.")
                    # for name, param in ebm_model_fr.named_parameters():
                    #     print(f"  EBM_FR Param {name}: requires_grad={param.requires_grad}, grad_fn (on param tensor): {param.grad_fn}, is_leaf: {param.is_leaf}")
                    break # Stop gradient steps if no grad_fn

                # Summing energy for a single gradient signal per batch item (across positions)
                # Or, apply gradient per position. Let's assume per position for now, then sum grads or average.
                # The current EBM training sums energies. For intervention, let's try to get grads per element.
                # If energy_fr is (N, 1), sum it to (1). If (N), sum to (1).
                # Or, use grad_outputs = torch.ones_like(energy_fr).
                
                # We need grads w.r.t current_slice_for_grad (batch, num_pos, hidden)
                # energy_fr is (batch*num_pos, 1) or (batch*num_pos)
                # We need to reshape grads back to (batch, num_pos, hidden)
                
                grads_fr_flat = torch.autograd.grad(
                    outputs=energy_fr.sum(), # Summing to get a single scalar for .grad()
                    inputs=current_slice_for_grad, # Grads w.r.t. the 3D slice
                    # grad_outputs=torch.ones_like(energy_fr, device=energy_fr.device), # if energy_fr is not scalar
                    retain_graph=True if num_gradient_steps > 1 else False, # Keep graph if more steps
                    allow_unused=False # Should not be unused
                )[0]

                if grads_fr_flat is not None:
                    # grads_fr_flat should have the same shape as current_slice_for_grad: (batch, num_pos, hidden)
                    norm_grads_fr = grads_fr_flat / (torch.norm(grads_fr_flat, dim=-1, keepdim=True) + 1e-8)
                    modified_slice = modified_slice - eta * norm_grads_fr # FR: decrease energy
                    # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] Applied FR modification. Grad norm: {torch.norm(grads_fr_flat).item()}")
                else:
                    # print(f"WARNING L{current_hook_layer_idx} Step {step+1}: grads_fr is None. Skipping FR modification.")
                    break

                # --- True Refusal (TR) EBM part (Orthogonal Steering) ---
                if ebm_model_tr and lambda_ebm_ortho != 0: # Ensuring this is at 16 spaces
                    if not current_slice_for_grad.requires_grad: # Ensure it still has grad if TR is first/only
                        current_slice_for_grad.requires_grad_(True) 
                    
                    ebm_model_tr.train() # Set to train mode for grad calculation
                    input_to_ebm_tr_reshaped = current_slice_for_grad.reshape(-1, hidden_dim) # Use potentially modified slice by FR
                    # Cast to EBM's weight dtype (float32) and ensure it's on the hook's specified device
                    input_to_ebm_tr_casted = input_to_ebm_tr_reshaped.to(device=device, dtype=torch.float32)
                    energy_tr = ebm_model_tr(input_to_ebm_tr_casted)
                    # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] EBM_TR energy: mean {energy_tr.mean().item()}, requires_grad {energy_tr.requires_grad}")
                    
                    if not original_ebm_tr_training_mode: ebm_model_tr.eval() # Restore

                    if energy_tr.grad_fn is None:
                        # print(f"WARNING L{current_hook_layer_idx} Step {step+1}: energy_tr has no grad_fn! Skipping grad step for TR.")
                        # No break here, FR might have worked.
                        pass
                    else:
                        grads_tr_flat = torch.autograd.grad(
                            outputs=energy_tr.sum(), # TR: increase energy (towards refusal)
                            inputs=current_slice_for_grad,
                            retain_graph=True if num_gradient_steps > 1 else False, # Keep graph if more steps
                            allow_unused=False
                        )[0]
                    
                        if grads_tr_flat is not None:
                            norm_grads_tr = grads_tr_flat / (torch.norm(grads_tr_flat, dim=-1, keepdim=True) + 1e-8)
                            # Project FR grads onto TR grads and subtract to orthogonalize
                            if grads_fr_flat is not None: # If FR step happened
                                proj_fr_on_tr = torch.sum(norm_grads_fr * norm_grads_tr, dim=-1, keepdim=True) * norm_grads_tr
                                ortho_grads_fr = norm_grads_fr - proj_fr_on_tr
                                # Re-apply FR with orthogonalized direction
                                modified_slice = modified_slice + eta * norm_grads_fr # Undo previous FR step
                                modified_slice = modified_slice - eta * ortho_grads_fr # Apply ortho FR step
                                # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] Applied Ortho FR modification.")
                            
                            # Apply TR modification (increase energy)
                            modified_slice = modified_slice + lambda_ebm_ortho * eta * norm_grads_tr 
                            # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] Applied TR modification. Grad norm: {torch.norm(grads_tr_flat).item()}")
                        else:
                            # print(f"WARNING L{current_hook_layer_idx} Step {step+1}: grads_tr is None. Skipping TR modification.")
                            pass # Don't break, FR might have worked.
            # End of gradient step loop
            # print(f"[EBM Hook Debug L{current_hook_layer_idx} Step {step+1}] Iteration norm of slice change: {torch.norm(modified_slice - act_slice_current_step).item()}")

        # Restore original EBM training modes if they were changed
        if ebm_model_fr.training != original_ebm_fr_training_mode:
            if original_ebm_fr_training_mode: ebm_model_fr.train()
            else: ebm_model_fr.eval()
        if ebm_model_tr and (ebm_model_tr.training != original_ebm_tr_training_mode):
            if original_ebm_tr_training_mode: ebm_model_tr.train()
            else: ebm_model_tr.eval()

        # Place the modified slice back into a copy of the original batch activations
        final_modified_activations_batch = original_activations_batch.clone()
        final_modified_activations_batch[:, actual_position_indices, :] = modified_slice.to(final_modified_activations_batch.dtype)
        
        modification_norm = torch.norm(final_modified_activations_batch - original_activations_batch).item()
        if num_gradient_steps > 0: # Only print if we attempted modification
             # print(f"[EBM Hook L{current_hook_layer_idx}] END. Norm of total modification on slice: {modification_norm}")
             pass # Final modification norm print commented out as requested

        return (final_modified_activations_batch,) + input_tensors[1:]

    # This print is for the factory function, to see what it's about to return
    print(f"[get_ebm_intervention_hook factory] Returning hook_fn of type: {type(hook_fn)} for layer {current_hook_layer_idx}")
    return hook_fn

# --- Helper to print hook structure (Conceptual) ---
def print_hook_structure(model: nn.Module, hooks_list: List[Tuple[nn.Module, str]]):
    # hooks_list is a list of (module_instance, hook_type_str e.g. "forward_pre" or "forward")
    # This is a conceptual helper, actual implementation would need more model introspection.
    pass
    # print("\nModel Hook Structure:")
    # for name, module in model.named_modules():
    #     has_hook = False
    #     hook_types = []
    #     for hooked_module, hook_type in hooks_list:
    #         if module is hooked_module: # Check for object identity
    #             has_hook = True
    #             hook_types.append(hook_type)
        
    #     if has_hook:
    #         print(f"Layer: {name} (Type: {type(module).__name__}) - HOOKED ({', '.join(hook_types)})")
    #     # else:
    #     #     print(f"Layer: {name} (Type: {type(module).__name__})")


# If you need to clear all hooks from a model (e.g., during testing or re-runs):
def clear_all_hooks(model: nn.Module):
    # This is a more forceful approach. PyTorch hook handles are usually managed by context managers or stored and removed.
    # Be cautious with directly manipulating internal hook dictionaries if not necessary.
    
    # For forward_pre_hooks
    for module in model.modules():
        if hasattr(module, '_forward_pre_hooks') and isinstance(module._forward_pre_hooks, dict):
            module._forward_pre_hooks.clear()
        # For forward_hooks
        if hasattr(module, '_forward_hooks') and isinstance(module._forward_hooks, dict):
            module._forward_hooks.clear()
        # For backward_hooks (less common for this use case but good to be aware of)
        if hasattr(module, '_backward_hooks') and isinstance(module._backward_hooks, dict):
            module._backward_hooks.clear()
