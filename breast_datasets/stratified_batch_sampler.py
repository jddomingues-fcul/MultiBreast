import numpy as np
from torch.utils.data import Sampler


class BootstrapingSampler(Sampler[int]):
    """Sampler that samples with replacement from the dataset, useful for bootstraping evaluation."""

    def __init__(self, targets: list, subset_size: int = 64):
        self.indices = np.array(range(len(targets)))
        self.subset_size = subset_size
        self.targets = np.array(targets)
        self.sample_indices = np.random.choice(
            self.indices, size=self.subset_size, replace=True
        )

    def __iter__(self):
        yield from self.sample_indices

    def __len__(self) -> int:
        return self.subset_size


class ConstantDiffuseSampler(Sampler[int]):
    """Sampler that always return the same indices, which are pretty diffuse across classes in the dataset to observe how the model is evolving over time."""

    def __init__(self, targets: list, batch_size: int):
        self.indices = np.array(range(len(targets)))
        self.targets = np.array(targets)
        self.batch_size = batch_size

        assert self.batch_size > 0, "Batch size must be greater than zero."
        self.batch_size = batch_size

        unique_classes = np.unique(self.targets)
        assert len(unique_classes) > 0, "No classes found in the dataset."

        self.result = []
        self.total_samples = max(self.batch_size, len(unique_classes))

        # Step 1: Initial allocation, 1 sample per class
        for cls in unique_classes:
            indices = np.where(self.targets == cls)[0]
            n_available = len(indices)
            if n_available > 0:
                selected_index = np.random.choice(indices, size=1, replace=False)
                self.result.append(selected_index[0])

        # Step 2: Fill the rest of the batch with random samples from the dataset if actually needed
        while len(self.result) < self.batch_size:
            random_index = np.random.choice(self.indices, size=1, replace=False)
            self.result.append(random_index[0])

    def __iter__(self):
        yield from self.result

    def __len__(self) -> int:
        return self.total_samples
