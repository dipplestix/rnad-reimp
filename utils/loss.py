import torch
import torch.special
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class HLGaussLoss(nn.Module):
    """HL-Gauss (Histogram Loss with Gaussian targets) for value function learning.
    
    Converts regression targets into soft classification targets using a Gaussian
    distribution over discretized bins. This approach has been shown to improve
    performance and scalability in deep RL.
    
    Implementation adapted from:
    "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
    
    Args:
        min_value: Minimum value of the support range.
        max_value: Maximum value of the support range.
        num_bins: Number of discrete bins for the value distribution.
        sigma: Standard deviation for the Gaussian distribution. If None, defaults
            to 0.75 * bin_width, which distributes probability mass to approximately
            6 neighboring bins (following σ/ς = 0.75 recommendation).
    
    Shape:
        - Input logits: (*, num_bins) where * is any number of batch dimensions
        - Target values: (*) matching the batch dimensions of logits
        - Output: scalar loss (mean over all elements)
    
    Note:
        Target values outside [min_value, max_value] are clipped to this range.
    """
    def __init__(self, 
    min_value: float, 
    max_value: float, 
    num_bins: int, 
    sigma: Optional[float] = None
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        bin_width = (max_value - min_value) / num_bins
        self.sigma = sigma if sigma is not None else 0.75 * bin_width
        self.register_buffer(
            'support',
            torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        # Clip target values to the valid range
        target = torch.clamp(target, self.min_value, self.max_value)
        
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1)) /
            (torch.sqrt(torch.tensor(2.0, device=target.device, dtype=target.dtype)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)