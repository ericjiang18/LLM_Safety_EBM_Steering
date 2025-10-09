import torch
import torch.nn as nn

class SimpleEBM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Define a simple MLP for the energy function
        # Example: A few linear layers with non-linearities
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Or nn.ReLU(), nn.Tanh(), etc.
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) # Output a single scalar energy value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input activation vector(s) of shape (batch_size, input_dim)
                              or (input_dim) if a single vector.
        Returns:
            torch.Tensor: Scalar energy value(s) of shape (batch_size, 1) or (1).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0) # Add batch dimension if single vector
        energy = self.network(x)
        return energy

class ComplexEBM(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = [1024, 512, 256], dropout_rate: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_layernorm = use_layernorm

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SiLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, 1)) # Output a single scalar energy value
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input activation vector(s) of shape (batch_size, input_dim)
                              or (input_dim) if a single vector.
        Returns:
            torch.Tensor: Scalar energy value(s) of shape (batch_size, 1) or (1).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0) # Add batch dimension if single vector
        energy = self.network(x)
        return energy

# You might want different EBMs for False Refusal (EBM_FR) and True Refusal (EBM_TR)
# or a multi-headed EBM.
# For now, we'll assume EBM_FR is the primary one.

def load_ebm_model(
    model_path: str, 
    input_dim: int, 
    device: str = 'cuda',
    ebm_architecture: str = 'simple', # New parameter: 'simple' or 'complex'
    simple_ebm_hidden_dim: int = 512, # For SimpleEBM
    complex_ebm_hidden_dims: list[int] = [1024, 512, 256], # For ComplexEBM
    complex_ebm_dropout_rate: float = 0.1, # For ComplexEBM
    complex_ebm_use_layernorm: bool = True # For ComplexEBM
):
    """
    Loads a trained EBM model from a checkpoint.
    Can load either SimpleEBM or ComplexEBM based on ebm_architecture.
    """
    if ebm_architecture == 'simple':
        model = SimpleEBM(input_dim, hidden_dim=simple_ebm_hidden_dim)
    elif ebm_architecture == 'complex':
        model = ComplexEBM(
            input_dim, 
            hidden_dims=complex_ebm_hidden_dims, 
            dropout_rate=complex_ebm_dropout_rate, 
            use_layernorm=complex_ebm_use_layernorm
        )
    else:
        print(f"Error: Unknown EBM architecture '{ebm_architecture}'. Cannot load model.")
        return None
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"EBM model ({ebm_architecture}) loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Warning: EBM model file not found at {model_path}. Returning an uninitialized model.")
        # Return the uninitialized model so it can be trained if force_retrain is True
        model.to(device) # Ensure it's on the right device even if not loaded
        return model 
    except Exception as e:
        print(f"Error loading EBM model from {model_path}: {e}")
        return None
    return model
