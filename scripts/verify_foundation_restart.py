"""Remote-only independent process check of the next canonical learning update."""

import random
import sys

import numpy as np
import torch

from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

if __name__ == "__main__":
    require_remote_cuda()
    state = LearningState.load(sys.argv[1])
    batch = state.external["next_batch"]
    indices = [state.external["sampler"].randrange(len(batch[0]))]
    state.metadata["next_indices"] = indices
    state.metadata["last_loss"] = state.step(*[v[indices] for v in batch])
    state.metadata["rng_after_update"] = [
        random.random(),
        float(np.random.random()),
        torch.rand(2).tolist(),
        torch.rand(2, device="cuda").tolist(),
    ]
    state.save(sys.argv[2])
