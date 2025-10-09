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
            self.handles.append(module.register_forward_pre_hook(hook_fn))
        for module, hook_fn in self.module_forward_hooks:
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
    forward_hooks_for_submodules = []

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
    for layer_idx in range(start_layer, model_base.model.config.num_hidden_layers):
        if hasattr(model_base, 'model_attn_modules') and layer_idx < len(model_base.model_attn_modules):
             output_ablation_hooks.append(
                 (model_base.model_attn_modules[layer_idx], get_direction_ablation_output_hook(direction=direction, coeff=coeff))
            )
        if hasattr(model_base, 'model_mlp_modules') and layer_idx < len(model_base.model_mlp_modules):
             output_ablation_hooks.append(
                 (model_base.model_mlp_modules[layer_idx], get_direction_ablation_output_hook(direction=direction, coeff=coeff))
            )
    # The provided `run_pipeline.py` expects `pre_hooks` and `fwd_hooks`.
    # Let's assume `pre_hooks` is empty from this function based on Appendix D (modifying output projections).
    return [], output_ablation_hooks # No pre-hooks from this specific wrapper, only output hooks for sub-modules

# --- New EBM Intervention Hook ---
def get_ebm_intervention_hook(
    ebm_model_fr: torch.nn.Module,
    target_layer_idx: int,
    current_hook_layer_idx: int,
    position_indices: List[int],
    eta: float,
    num_gradient_steps: int = 1,
    ebm_model_tr: Optional[torch.nn.Module] = None,
    lambda_ebm_ortho: float = 0.0,
    device: str = 'cuda' # Add device for tensor operations
) -> Callable:
    """
    Returns a forward pre-hook that modifies activations using EBM gradients.
    Assumes EBM models are already on the correct device.
    """
    def hook_fn(module: torch.nn.Module, input_tensors: Tuple[torch.Tensor]) -> Optional[Tuple[torch.Tensor]]:
        if current_hook_layer_idx != target_layer_idx:
            return None # Let original input pass through if not the target layer

        original_activations_batch = input_tensors[0]
        modified_activations_batch = original_activations_batch.clone()

        for pos_idx_relative in position_indices:
            actual_pos_idx = original_activations_batch.shape[1] + pos_idx_relative if pos_idx_relative < 0 else pos_idx_relative
            
            if not (0 <= actual_pos_idx < original_activations_batch.shape[1]):
                continue # Skip if position is out of bounds

            act_slice_current_step = modified_activations_batch[:, actual_pos_idx, :].clone().detach().requires_grad_(True)

            for step in range(num_gradient_steps):
                # Use an explicit gradient-enabled context for EBM operations
                with torch.enable_grad():
                    current_input_for_grad_step = act_slice_current_step.clone()
                    current_input_for_grad_step.requires_grad_(True)
                    
                    ebm_model_fr.eval()
                    if ebm_model_tr: ebm_model_tr.eval()

                    ebm_dtype = next(ebm_model_fr.parameters()).dtype
                    input_to_ebm_fr = current_input_for_grad_step.to(device=device, dtype=ebm_dtype)
                    # If .to() created a new tensor, it should inherit requires_grad if source did.
                    # If it's the same tensor, requires_grad is already set.

                    energy_fr = ebm_model_fr(input_to_ebm_fr)
                    scalar_energy_fr = energy_fr.sum()
                    
                    if input_to_ebm_fr.grad is not None:
                        input_to_ebm_fr.grad.zero_()

                    retain_graph_flag = bool(ebm_model_tr is not None or step < num_gradient_steps - 1)
                    grad_fr = torch.autograd.grad(
                        outputs=scalar_energy_fr, 
                        inputs=input_to_ebm_fr, 
                        retain_graph=retain_graph_flag, 
                        create_graph=False
                    )[0]
                
                # grad_fr is now computed. The rest of the update can be outside enable_grad if preferred,
                # but the inputs to the arithmetic should be from the grad-enabled block or handled carefully.
                update_direction = -grad_fr

                if ebm_model_tr is not None and lambda_ebm_ortho > 0.0:
                    with torch.enable_grad(): # Separate context for TR gradient if needed
                        # current_input_for_grad_step is already set up with requires_grad
                        ebm_tr_dtype = next(ebm_model_tr.parameters()).dtype
                        input_to_ebm_tr = current_input_for_grad_step.to(device=device, dtype=ebm_tr_dtype)
                        
                        energy_tr = ebm_model_tr(input_to_ebm_tr).sum()
                        
                        if input_to_ebm_tr.grad is not None:
                             input_to_ebm_tr.grad.zero_()

                        grad_tr = torch.autograd.grad(
                            outputs=energy_tr, 
                            inputs=input_to_ebm_tr,
                            retain_graph=bool(step < num_gradient_steps - 1),
                            create_graph=False
                        )[0]
                    
                    # Orthogonalization logic using grad_tr and update_direction would go here
                    pass 
                
                # Update act_slice_current_step. Ensure devices match for the arithmetic operation.
                act_slice_current_step = (input_to_ebm_fr.to(update_direction.device) + eta * update_direction).detach()

            modified_activations_batch[:, actual_pos_idx, :] = act_slice_current_step.to(original_activations_batch.device, dtype=original_activations_batch.dtype)
            
        return (modified_activations_batch,) + input_tensors[1:]
    return hook_fn
