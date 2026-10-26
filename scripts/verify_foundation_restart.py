"""Remote-only independent process check of the next canonical learning update."""

import sys

from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

if __name__ == "__main__":
    require_remote_cuda()
    state = LearningState.load(sys.argv[1])
    state.metadata["last_loss"] = state.step(*state.external["next_batch"])
    state.save(sys.argv[2])
