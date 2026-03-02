# TODO[REFACTOR]: integrate into cache later (too high integration costs due to the cache's poor design).
from collections import defaultdict
import math
from typing import TYPE_CHECKING

from xfuser.core.distributed import get_runtime_state

if TYPE_CHECKING:
    from torch import Tensor


class DirectReuse:
    # Direct residual re-use without correction

    def __init__(self):
        # {tensor name: [{order: tensor value}]}, where the position in the list is the patch id
        self.cache: defaultdict[str, list[dict[int, "Tensor"]]] = defaultdict(lambda: [{}])
        self.patch_mode = False

    def _check_patch_mode(self) -> int:
        patch_mode = get_runtime_state().patch_mode
        if self.patch_mode != patch_mode:
            if patch_mode:  # split cache for async execution
                for name, val in self.cache.items():
                    self.cache[name] = [
                        {order: split_t}
                        for order, tensor in val[0].items()
                        for split_t in tensor.chunk(get_runtime_state().num_pipeline_patch, dim=-2)
                    ]
            else:  # reset cache for a new sample
                # FIXME: reset fails in fully sync mode
                self.cache = defaultdict(lambda: [{}])
            self.patch_mode = patch_mode
        return get_runtime_state().pipeline_patch_idx

    def update(self, name, value, **kwargs):
        patch_id = self._check_patch_mode()
        self.cache[name][patch_id][0] = value

    def forecast(self, name, **kwargs):
        patch_id = self._check_patch_mode()
        return self.cache[name][patch_id][0]


class TaylorSeer(DirectReuse):
    def __init__(self, max_order: int = 3):
        super().__init__()
        self.order = max_order
        self.current_step, self.last_non_approximated_step = [-1], [-1]

    def _check_patch_mode(self) -> int:
        patch_mode = get_runtime_state().patch_mode
        if self.patch_mode != patch_mode:
            if patch_mode:  # split cache for async execution
                for name, val in self.cache.items():
                    self.cache[name] = [
                        {order: split_t}
                        for order, tensor in val[0].items()
                        for split_t in tensor.chunk(get_runtime_state().num_pipeline_patch, dim=-2)
                    ]
                self.current_step = self.current_step * get_runtime_state().num_pipeline_patch
                self.last_non_approximated_step = (
                    self.last_non_approximated_step * get_runtime_state().num_pipeline_patch
                )
            else:  # reset cache for a new sample
                # FIXME: reset fails in fully sync mode
                self.cache = defaultdict(lambda: [{}])
                self.current_step, self.last_non_approximated_step = [-1], [-1]
            self.patch_mode = patch_mode
        return get_runtime_state().pipeline_patch_idx

    def update(self, name, value, **kwargs):
        patch_id = self._check_patch_mode()

        self.current_step[patch_id] += 1
        distance = self.current_step[patch_id] - self.last_non_approximated_step[patch_id]

        updated_cache = {0: value}
        cache = self.cache[name][patch_id]
        for i in range(self.order):
            if cache.get(i, None) is not None:
                updated_cache[i + 1] = (updated_cache[i] - cache[i]) / distance

        self.cache[name][patch_id] = updated_cache
        self.last_non_approximated_step[patch_id] = self.current_step[patch_id]

    def forecast(self, name, **kwargs):
        patch_id = self._check_patch_mode()

        self.current_step[patch_id] += 1
        distance = self.current_step[patch_id] - self.last_non_approximated_step[patch_id]

        output = 0
        cache = self.cache[name][patch_id]
        for i in range(len(cache)):
            output += (1 / math.factorial(i)) * cache[i] * (distance**i)
        return output
