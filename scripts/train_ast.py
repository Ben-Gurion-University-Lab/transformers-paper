"""Fine-tune an AST classifier on prepared respiratory audio features."""

import os
import torch
import torch.multiprocessing as mp
from datetime import datetime, timedelta
import time
import yaml
import wandb
from transformers import (
    ASTForAudioClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from sklearn.metrics import accuracy_score, f1_score
from breathe_transformers.ast import (
    ASTDatasetConfig,
    ASTDataset,
    custom_collate,
)


def load_yaml_config(config_path: str) -> dict:
    """Load and parse YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"[DEBUG] Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        raise


class WandbMetricsCallback(TrainerCallback):
    """Custom callback for detailed wandb logging."""

    def __init__(self, trainer: Trainer):
        """Initialize callback state for one trainer."""
        self.trainer = trainer
        self.start_time = time.time()
        self.best_metric = 0.0
        self.best_accuracy = 0.0
        self.training_tracker = {"train_step": 0, "eval_step": 0}

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log training metrics after each step."""
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            logs = state.log_history[-1] if state.log_history else {}
            current_lr = (
                self.trainer.optimizer.param_groups[0]["lr"]
                if self.trainer.optimizer
                else 0
            )

            # Get training metrics
            predictions = kwargs.get("predictions")
            labels = kwargs.get("labels")
            if predictions is not None and labels is not None:
                predictions = predictions.argmax(-1)
                train_accuracy = (predictions == labels).float().mean().item()
            else:
                train_accuracy = logs.get("accuracy", 0)

            step_metrics = {
                "train/step": self.training_tracker["train_step"],
                "train/loss": logs.get("loss", 0),
                "train/accuracy": train_accuracy,
                "train/learning_rate": current_lr,
            }
            wandb.log(step_metrics)  # ty: ignore
            self.training_tracker["train_step"] += 1

    def on_evaluate(  # ty: ignore
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ):
        """Log evaluation metrics."""
        eval_metrics = {
            "valid/step": self.training_tracker["eval_step"],
            "valid/loss": metrics.get("eval_loss", 0),
            "valid/accuracy": metrics.get("eval_accuracy", 0),
            "valid/f1": metrics.get("eval_f1", 0),
        }
        wandb.log(eval_metrics)  # ty: ignore
        self.training_tracker["eval_step"] += 1

        # Track best metrics
        current_metric = metrics.get("eval_f1", 0)
        current_accuracy = metrics.get("eval_accuracy", 0)

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            wandb.log({"epoch/best_f1": self.best_metric})  # ty: ignore

        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            wandb.log({"epoch/best_accuracy": self.best_accuracy})  # ty: ignore

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log epoch-level metrics."""
        if state.epoch is not None:
            current_lr = (
                self.trainer.optimizer.param_groups[0]["lr"]
                if self.trainer.optimizer
                else 0
            )

            # Safely get the last metrics from log history
            last_loss = 0
            last_accuracy = 0
            if state.log_history:
                for log in reversed(state.log_history):
                    if "loss" in log and last_loss == 0:
                        last_loss = log["loss"]
                    if "accuracy" in log and last_accuracy == 0:
                        last_accuracy = log["accuracy"]
                    if last_loss != 0 and last_accuracy != 0:
                        break

            epoch_metrics = {
                "epoch": int(state.epoch),
                "epoch/train_loss": last_loss,
                "epoch/train_accuracy": last_accuracy,
                "epoch/learning_rate": current_lr,
                "epoch/train_time": time.time() - self.start_time,
            }
            wandb.log(epoch_metrics)  # ty: ignore

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log final training metrics."""
        wandb.log(  # ty: ignore
            {
                "training/total_time": time.time() - self.start_time,
                "training/best_f1": self.best_metric,
                "training/best_accuracy": self.best_accuracy,
                "training/total_steps": state.global_step,
            }
        )


def compute_metrics(eval_pred):
    """Compute classification metrics for AST evaluation."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = predictions.argmax(-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def get_run_name(base_name: str | None = None) -> str:
    """Return an explicit run name or generate a unique run_name from timestamp."""
    if base_name:
        return base_name
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def train_ast(
    train_dataset: ASTDataset,
    eval_dataset: ASTDataset | None = None,
    config_path: str = "hparams/ast_asthma.yaml",  # 5-second paper config
    output_dir: str = "temp/ast_model_5sec",  # 5-second output dir
    use_wandb: bool = True,
    project_name: str = "ast-finetuning-5sec",  # 5-second project name
    run_name: str | None = None,
):
    """Train AST model on preprocessed 5-second dataset."""
    # Load configuration
    print("\n[DEBUG] Loading training configuration...")
    config = load_yaml_config(config_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    print(f"\n[DEBUG] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
        print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[DEBUG] Device name: {torch.cuda.get_device_name(0)}")

    # Generate unique run name and output directory
    run_name = get_run_name(run_name)
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"\n[DEBUG] Run name: {run_name}")
    print(f"[DEBUG] Output directory: {run_output_dir}")

    # Print dataset information
    print("\n[DEBUG] Dataset configuration:")
    print(f"[DEBUG] Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"[DEBUG] Evaluation dataset size: {len(eval_dataset)}")

    # Print audio length information
    audio_length = config.get("audio_length", 10)  # Default to 10 if not specified
    print(f"[DEBUG] Audio clip length: {audio_length} seconds")

    if use_wandb:
        print("\n[DEBUG] Initializing WandB...")
        wandb_config = {
            # Model Configuration
            "model": {
                "name": config["model_name"],
                "num_labels": train_dataset.num_labels,
                "labels": train_dataset.id2label,
            },
            # Training Configuration
            "training": {
                **config,  # Include all config parameters
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset) if eval_dataset else None,
                "audio_length": audio_length,  # Explicitly log audio length
            },
            # System Information
            "system": {
                "device": device.type,
                "cuda_available": torch.cuda.is_available(),
                "num_gpus": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
            },
        }
        wandb.init(project=project_name, name=run_name, config=wandb_config)  # ty: ignore

    # Initialize model
    print("\n[DEBUG] Initializing model...")
    model = ASTForAudioClassification.from_pretrained(
        config["model_name"],
        num_labels=train_dataset.num_labels,
        label2id=train_dataset.label2id,
        id2label=train_dataset.id2label,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Print model parameters
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n[DEBUG] Model parameters:")
    print(f"[DEBUG] Total parameters: {n_parameters:,}")
    print(f"[DEBUG] Trainable parameters: {n_trainable:,}")

    # Training arguments
    print("\n[DEBUG] Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=1,  # Log every step
        logging_first_step=True,  # Log the first step
        eval_steps=50 if eval_dataset else None,  # Evaluate more frequently
        eval_strategy="steps" if eval_dataset else "no",
        save_steps=100,
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="f1" if eval_dataset else None,
        report_to="wandb" if use_wandb else "none",
        no_cuda=not torch.cuda.is_available(),
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=config["pin_memory"],
        run_name=run_name,
        max_grad_norm=config["max_grad_norm"],
        fp16=config["fp16"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type=config["scheduler_type"],
    )

    print("\n[DEBUG] Training configuration:")
    print(f"[DEBUG] Number of epochs: {config['num_train_epochs']}")
    print(f"[DEBUG] Batch size: {config['per_device_train_batch_size']}")
    print(f"[DEBUG] Learning rate: {config['learning_rate']}")
    print(
        f"[DEBUG] Gradient accumulation steps: {config['gradient_accumulation_steps']}"
    )
    print(
        f"[DEBUG] Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}"
    )
    steps_per_epoch = len(train_dataset) / (
        config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    )
    total_steps = steps_per_epoch * config["num_train_epochs"]
    print(f"[DEBUG] Steps per epoch: {steps_per_epoch:.2f}")
    print(f"[DEBUG] Total training steps: {total_steps:.2f}")
    print("[DEBUG] Logging frequency: Every step")
    print("[DEBUG] Evaluation frequency: Every 50 steps")

    # Initialize trainer with custom data collator and callbacks
    print("\n[DEBUG] Initializing trainer...")

    # Add progress printing callback
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 1 == 0:  # Print every step
                print(f"\r[DEBUG] Step {state.global_step}/{int(total_steps)}", end="")
                if "loss" in kwargs:
                    print(f" - Loss: {kwargs['loss']:.4f}", end="")
                if "metrics" in kwargs and "accuracy" in kwargs["metrics"]:
                    print(f" - Accuracy: {kwargs['metrics']['accuracy']:.4f}", end="")

    callbacks: list[EarlyStoppingCallback | ProgressCallback | None] = [
        (
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"]
            )
            if eval_dataset
            else None
        ),
    ]
    callbacks = [cb for cb in callbacks if cb is not None]  # Remove None values

    callbacks.append(ProgressCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_collate,
        callbacks=callbacks,
    )

    # Add wandb callback if enabled
    if use_wandb:
        print("[DEBUG] Adding WandB callback...")
        wandb_callback = WandbMetricsCallback(trainer)
        trainer.add_callback(wandb_callback)

    # Train
    print("\n[DEBUG] Starting training...")
    trainer.train()
    print("\n[DEBUG] Training completed!")

    # Save final model with run-specific name
    final_model_path = os.path.join(run_output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\n[DEBUG] Final model saved at: {final_model_path}")

    # Save label mappings and training metadata
    training_metadata = {
        "label_mapping": {
            "label2id": train_dataset.label2id,
            "id2label": train_dataset.id2label,
        },
        "training_info": {
            "total_time": time.time() - start_time,
            "config": config,
            "final_learning_rate": trainer.optimizer.param_groups[0]["lr"],  # ty: ignore
            "audio_length": audio_length,  # Save audio length in metadata
        },
        "model_config": model.config.to_dict(),
    }
    torch.save(
        training_metadata, os.path.join(final_model_path, "training_metadata.pt")
    )
    print("[DEBUG] Training metadata saved")

    if use_wandb:
        # Log final metrics and finish run
        wandb.log(  # ty: ignore
            {
                "final/total_time": time.time() - start_time,
                "final/model_path": final_model_path,
            }
        )
        wandb.finish()  # ty: ignore
        print("[DEBUG] WandB logging completed")

    total_time = time.time() - start_time
    print(f"\n[DEBUG] Total training time: {str(timedelta(seconds=int(total_time)))}")

    return final_model_path


if __name__ == "__main__":
    import argparse

    # Set multiprocessing start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train AST model on 5-second clips")
    parser.add_argument("--train_features_dir", type=str, required=True)
    parser.add_argument("--train_metadata", type=str, required=True)
    parser.add_argument("--eval_features_dir", type=str)
    parser.add_argument("--eval_metadata", type=str)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_dir", type=str, default="temp/ast_model_5sec")
    parser.add_argument(
        "--config_path",
        type=str,
        default="hparams/ast_asthma.yaml",  # 5-second paper config
        help="Path to YAML configuration file",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--project_name", type=str, default="ast-finetuning-5sec"
    )  # 5-second project name
    parser.add_argument("--run_name", type=str)

    args = parser.parse_args()

    # Setup device "cuda", "mps", or "cpu"
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize datasets
    train_config = ASTDatasetConfig(
        features_dir=args.train_features_dir,
        metadata_path=args.train_metadata,
        label_column=args.label_column,
    )
    train_dataset = ASTDataset(train_config)

    eval_dataset = None
    if args.eval_features_dir and args.eval_metadata:
        eval_config = ASTDatasetConfig(
            features_dir=args.eval_features_dir,
            metadata_path=args.eval_metadata,
            label_column=args.label_column,
        )
        eval_dataset = ASTDataset(eval_config)

    # Train model
    final_model_path = train_ast(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config_path=args.config_path,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.run_name,
    )

    print(f"Training completed. Final model saved at: {final_model_path}")
