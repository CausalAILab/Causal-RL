# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/abstract_teacher.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
from abc import ABC, abstractmethod


class AbstractTeacher(ABC):
    """
    A curriculum teacher proposes training contexts via `sample()`. Teachers that adapt to the
    learner are updated by their environment wrapper's `done_callback` (see `BaseWrapper`) with
    per-episode feedback: the context and its discounted return (ALP-GMM; PLR also gets the
    learner's value estimates; CurrOT is updated in batches of EP_PER_UPDATE episodes) or its
    success flag `info['success']` (GoalGAN).
    """

    @abstractmethod
    def sample(self):
        pass
