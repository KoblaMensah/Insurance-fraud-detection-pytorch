"""
PyTorch Dataset for Insurance Claims

Wraps preprocessed numpy arrays into a PyTorch Dataset for use with
DataLoader. This separation keeps data handling clean and follows
PyTorch best practices.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class ClaimsDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed claims data."""

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features), already scaled.
        y : np.ndarray
            Binary labels (n_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(data_dict, batch_size=64):
    """
    Create train and test DataLoaders from preprocessed data.

    Parameters
    ----------
    data_dict : dict
        Output from DataPreprocessor.fit_transform().
    batch_size : int
        Batch size for training.

    Returns
    -------
    tuple of (train_loader, test_loader)
    """
    train_dataset = ClaimsDataset(data_dict["X_train"], data_dict["y_train"])
    test_dataset = ClaimsDataset(data_dict["X_test"], data_dict["y_test"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"DataLoaders created: {len(train_loader)} train batches, "
          f"{len(test_loader)} test batches (batch_size={batch_size})")

    return train_loader, test_loader
