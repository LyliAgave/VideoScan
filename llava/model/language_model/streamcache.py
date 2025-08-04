from transformers.cache_utils import DynamicCache
import torch

class StreamCache(DynamicCache):
    def __init__(self, past_key_values) -> None:
        super().__init__()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                self.key_cache.append(past_key_values[layer_idx][0])
                self.value_cache.append(past_key_values[layer_idx][1])

    def set_autograd(self):
        """
        set auto grad
        """
        for layer_idx in range(len(self.key_cache)):
            if isinstance(self.key_cache[layer_idx], torch.Tensor):
                self.key_cache[layer_idx].requires_grad_()

        for layer_idx in range(len(self.value_cache)):
            if isinstance(self.value_cache[layer_idx], torch.Tensor):
                self.value_cache[layer_idx].requires_grad_()