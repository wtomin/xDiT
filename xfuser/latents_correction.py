# TODO[REFACTOR]: integrate to `xfuser/model_executor/cache` later
import math

from xfuser.core.distributed import get_runtime_state


class PatchReuse:
    # Direct patch re-use without correction
    # TODO: don't simply re-use - calculate and add residuals instead!

    def __init__(self):
        self.cache = [{}]

    def update(self, feature, patch_id: int = 0, **kwargs):
        self.cache[patch_id][0] = feature

    def forecast(self, patch_id: int = 0, **kwargs):
        return self.cache[patch_id][0]

    def split_patches(self):
        assert len(self.cache) == 1, "The latents have already been split into patches."
        self.cache = [
            {k: split_v}
            for k, v in self.cache[0].items()
            for split_v in v.chunk(get_runtime_state().num_pipeline_patch, dim=-2)
        ]


class TaylorSeer(PatchReuse):
    def __init__(self, max_order: int = 3):
        super().__init__()
        self.max_order = max_order

    def update(self, feature, patch_id: int = 0, distance: int = 1):
        updated_cache = {0: feature}
        cache = self.cache[patch_id]
        for i in range(self.max_order):
            if cache.get(i, None) is not None:
                updated_cache[i + 1] = (updated_cache[i] - cache[i]) / distance

        self.cache[patch_id] = updated_cache

    def forecast(self, patch_id: int = 0, distance: int = 1):
        output = 0
        cache = self.cache[patch_id]
        for i in range(len(cache)):
            output += (1 / math.factorial(i)) * cache[i] * (distance**i)
        return output
