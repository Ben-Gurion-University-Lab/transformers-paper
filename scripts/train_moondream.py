"""Fine-tune a Moondream2 LoRA adapter on prepared respiratory images."""

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model  # ty: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    dynamic_module_utils,
    get_cosine_schedule_with_warmup,
)

from breathe_transformers.datasets import RespiratoryDataset
from breathe_transformers.moondream import (
    BASE_MODEL,
    MODEL_REVISION,
    _is_hf_snapshot_cached,
    disable_transformers_adapter_auto_detection,
    patch_dynamic_cache_compat,
    patch_generation_mixin,
)
from breathe_transformers.torch_utils import get_default_device

ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Moondream2 LoRA adapter on prepared data.json files."
    )
    parser.add_argument("--train_data_json", type=str, required=True)
    parser.add_argument("--train_images_folder", type=str, required=True)
    parser.add_argument("--eval_data_json", type=str)
    parser.add_argument("--eval_images_folder", type=str)
    parser.add_argument("--output_dir", type=str, default="models/moondream_finetuned")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--eval_batch_size", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=64)
    # Changed lora alpha to avoid overfitting; previous value was 32.
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--eval_fraction", type=float, default=0.025)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="moondream-breathe-asthma")
    return parser.parse_args()


def make_collate_fn(tokenizer):
    """Build the Moondream2 collate function for image QA samples."""

    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        labels_acc = []
        tokens_acc = []

        for sample in batch:
            tokens = [tokenizer.bos_token_id]
            labels = [-100] * (IMG_TOKENS + 1)

            for qa in sample["qa"]:
                question_tokens = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                tokens.extend(question_tokens)
                labels.extend([-100] * len(question_tokens))

                answer_tokens = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False,
                ).input_ids
                tokens.extend(answer_tokens)
                labels.extend(answer_tokens)

            tokens_acc.append(tokens)
            labels_acc.append(labels)

        max_len = max(len(labels) for labels in labels_acc)
        attn_mask_acc = []
        for idx, labels in enumerate(labels_acc):
            current_len = len(labels)
            pad_len = max_len - current_len
            labels.extend([-100] * pad_len)
            tokens_acc[idx].extend([tokenizer.eos_token_id] * pad_len)
            attn_mask_acc.append([1] * current_len + [0] * pad_len)

        return (
            images,  # Pass images as PIL images
            torch.tensor(tokens_acc, dtype=torch.long),
            torch.tensor(labels_acc, dtype=torch.long),
            torch.tensor(attn_mask_acc, dtype=torch.bool),
        )

    return collate_fn


def load_base_model(args: argparse.Namespace):
    """Load the base Moondream2 model and tokenizer for training."""
    device = args.device or get_default_device()
    dtype = torch.float32 if device in {"cpu", "mps"} else torch.bfloat16
    cache_dir = args.cache_dir or os.path.join(os.getcwd(), "temp", "cache")
    modules_cache = os.path.join(cache_dir, "modules")
    os.makedirs(modules_cache, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = modules_cache
    dynamic_module_utils.HF_MODULES_CACHE = modules_cache
    local_files_only = _is_hf_snapshot_cached(cache_dir, args.base_model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    with disable_transformers_adapter_auto_detection():
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=args.revision,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if device == "cuda" else None,
            torch_dtype=dtype,
            device_map={"": device},
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    patch_generation_mixin(model)
    patch_dynamic_cache_compat()
    return model, tokenizer, device


def compute_loss(model, batch, device: str):
    """Compute one Moondream2 training loss."""
    images, tokens, labels, attn_mask = batch
    # Images are a list of PIL images; pass them directly to the VisionEncoder
    tokens = tokens.to(device)
    labels = labels.to(device)
    attn_mask = attn_mask.to(device)

    with torch.no_grad():
        img_embeddings = model.vision_encoder(images)

    tok_embeddings = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embeddings[:, :1, :], img_embeddings, tok_embeddings[:, 1:, :]),
        dim=1,
    )
    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )
    return outputs.loss


def main() -> None:
    """Run Moondream2 LoRA fine-tuning."""
    args = parse_args()
    model, tokenizer, device = load_base_model(args)

    # Apply LoRA.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["proj", "fc1", "fc2", "Wqkv", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # modules_to_save=['lm_head','embd'],  # won't work with the trainer unless using a hf trainer, not custom.
    )
    model = get_peft_model(model, lora_config)

    train_dataset = RespiratoryDataset(
        json_path=args.train_data_json,
        images_folder=args.train_images_folder,
        split="train",
    )
    eval_dataset = None
    if args.eval_data_json and args.eval_images_folder:
        eval_dataset = RespiratoryDataset(
            json_path=args.eval_data_json,
            images_folder=args.eval_images_folder,
            split="test",
        )

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn,
        )

    if device == "cuda":
        from bitsandbytes.optim import Adam8bit  # ty: ignore
    else:
        from torch.optim import Adam as Adam8bit

    model.text_model.train()
    model.text_model.transformer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Fine-tune the LoRA parameters.
    lora_params = [param for param in model.parameters() if param.requires_grad]

    optimizer = Adam8bit(
        [{"params": lora_params, "weight_decay": args.weight_decay}],
        lr=args.learning_rate,  # * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )
    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps

    # Cosine learning rate schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    eval_steps = max(
        1, int(total_steps * args.eval_fraction)
    )  # this is to cast the steps to int

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))  # ty: ignore

    step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            step += 1
            loss = compute_loss(model, batch, device)
            loss.backward()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            metrics = {"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            if eval_loader is not None and step % eval_steps == 0:
                # Calculate validation loss
                val_loss = 0.0
                for eval_batch in tqdm(eval_loader, desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(model, eval_batch, device).item()
                metrics["loss/val"] = val_loss / len(eval_loader)

            if args.use_wandb:
                wandb.log(metrics)  # ty: ignore

    if args.use_wandb:
        wandb.finish()  # ty: ignore

    os.makedirs(args.output_dir, exist_ok=True)

    # Save the LoRA adapters
    model.save_pretrained(
        os.path.join(args.output_dir, "adapter"),
        save_embedding_layers=False,
    )


if __name__ == "__main__":
    main()
