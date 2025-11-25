import torch
import torch.nn as nn

class NormModulator(nn.Module):
    """
    Modulates normalization layer parameters in a fixed network
    using a learned embedding.
    """
    def __init__(self, base_model, embedding_dim):
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim

        self.transform = nn.Linear(embedding_dim, embedding_dim)
        self.transform_act = lambda x: x * torch.sigmoid(x) # swish

        self.transform_2 = nn.Linear(embedding_dim, embedding_dim)
        
        # Collect all normalization layers
        self.norm_layers = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.norm_layers.append((name.replace(".", "_"), module))

        # Learn mappings from embedding to scale and bias for each norm layer
        self.modulators = nn.ModuleDict()
        for name, norm in self.norm_layers:
            if isinstance(norm, nn.LayerNorm):
                linear = nn.Linear(embedding_dim, 2 * norm.normalized_shape[0])
            else:
                linear = nn.Linear(embedding_dim, 2 * norm.num_features)
            # 🔸 Initialize all weights and biases to zero
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.modulators[name] = linear

    def forward(self, x, embedding):
        embedding = self.transform_act(self.transform(embedding))
        embedding = self.transform_act(self.transform_2(embedding))
        # Compute all modulation parameters
        mod_params = {}
        for name, norm in self.norm_layers:
            params = self.modulators[name](embedding)  # [B, 2*C]
            gamma, beta = params.chunk(2, dim=-1)
            mod_params[name] = (gamma, beta)

        # Apply network with modified normalization
        # We'll need to hook forward passes of each normalization layer
        hooks = []

        def make_hook(name, norm):
            def hook(module, input, output):
                gamma, beta = mod_params[name]
                if len(output.shape) == 4:
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # For 2D norms
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                else:
                    gamma = gamma.unsqueeze(0)  # For 2D norms
                    beta = beta.unsqueeze(0)                    
                return output * torch.exp(gamma) + beta
            return norm.register_forward_hook(hook)

        # Register hooks
        for name, norm in self.norm_layers:
            hooks.append(make_hook(name, norm))

        # Forward through base model
        out = self.base_model(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        return out