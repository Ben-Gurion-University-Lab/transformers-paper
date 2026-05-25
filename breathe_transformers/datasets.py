"""Dataset classes for prepared transformer respiratory sound data."""

import json
import os
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ASTDatasetConfig:
    """Configuration for loading AST features and metadata."""

    features_dir: str
    metadata_path: str
    label_column: str = "label"
    feature_column: str = "file_path"
    num_labels: int | None = None


class ASTDataset(Dataset):
    """Dataset backed by prepared AST feature tensors."""

    def __init__(self, config: ASTDatasetConfig):
        """Initialize the dataset from an AST dataset config."""
        self.features_dir = config.features_dir
        self.metadata = pd.read_csv(config.metadata_path)
        self.label_column = config.label_column
        self.feature_column = config.feature_column

        # Create label mapping
        unique_labels = sorted(self.metadata[self.label_column].unique())
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def __len__(self):
        """Return the number of metadata rows."""
        return len(self.metadata)

    def __getitem__(self, index):
        """Return the feature tensor and label for one row."""
        row = self.metadata.iloc[index]
        feature_path = os.path.join(self.features_dir, row[self.feature_column])
        features = torch.load(feature_path, map_location="cpu")

        features, debug_info = self.process_features(features, file_path=feature_path)
        label = self.label2id[row[self.label_column]]

        return {"input_values": features, "labels": label, "debug_info": debug_info}

    @property
    def num_labels(self):
        """Return the number of unique labels in the dataset."""
        return len(self.label2id)

    def process_features(
        self, features: torch.Tensor, file_path: str
    ) -> tuple[torch.Tensor, list[str]]:
        """Process features and track transformations."""
        debug_info = [f"Original shape: {features.shape}"]

        try:
            if features.dim() == 6:  # [32, 1, 1, 1, 1024, 128]
                features = features.squeeze()
                debug_info.append(f"After squeeze from 6D: {features.shape}")
                if features.dim() > 4:
                    features = features[0]
                    debug_info.append(f"After taking first item: {features.shape}")
            elif features.dim() == 5:  # [B, 1, T, 1, F]
                features = features.squeeze()
                debug_info.append(f"After squeeze from 5D: {features.shape}")
                features = features.squeeze()
                debug_info.append(f"After second squeeze: {features.shape}")
            elif features.dim() == 3:  # [1, T, F]
                features = features.unsqueeze(1)
                debug_info.append(f"After adding channel dimension: {features.shape}")

            if features.dim() == 4 and features.shape[0] != 1:
                features = features[0]
                debug_info.append(f"After handling 4D tensor: {features.shape}")

            return features, debug_info
        except Exception as e:
            debug_info.append(f"Error during processing: {str(e)}")
            raise ValueError(
                f"Error processing {file_path}: {str(e)}\nDebug info:\n"
                + "\n".join(debug_info)
            )


class RespiratoryDataset(Dataset):
    """Moondream2 image-question-answer dataset."""

    def __init__(self, json_path: str, images_folder: str, split: str = "train"):
        """Load a prepared Moondream2 data.json file."""
        with open(json_path) as file:
            self.data = json.load(file)
        self.images_folder = images_folder
        self.split = split

    def __len__(self):
        """Return the number of dataset examples."""
        return len(self.data)

    def __getitem__(self, index):
        """Return one PIL image and question-answer pair."""
        sample = self.data[index]
        image_path = os.path.join(self.images_folder, sample["image"])
        image = Image.open(image_path).convert("RGB")
        question = sample["conversations"][0]["value"].replace("<image>\n", "")
        answer = sample["conversations"][1]["value"]

        return {
            "id": sample.get("id", str(index)),
            "image": image,
            "qa": [{"question": question, "answer": answer}],
        }
