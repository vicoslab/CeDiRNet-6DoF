class LockableSeedRandomAccess:
    """Interface for datasets that can preserve hard-sample augmentation seeds."""

    def lock_samples_seed(self, index_list):
        raise NotImplementedError()
