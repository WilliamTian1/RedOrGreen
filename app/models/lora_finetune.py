"""Optional LoRA fine-tuning for FinBERT.

This module provides a minimal stub that gracefully skips if PEFT/accelerate or GPU
are unavailable. It demonstrates where and how LoRA adaptation would be wired.
"""

from __future__ import annotations

from typing import Optional

from app.utils.logging import get_logger


logger = get_logger(__name__)


def maybe_lora_finetune(enabled: bool, model_id: str = "yiyanghkust/finbert-tone") -> Optional[str]:
    """Run a tiny LoRA fine-tune if enabled and environment supports it.

    Returns the path to a saved adapter, or None if skipped.
    """
    if not enabled:
        logger.info("LoRA fine-tune disabled via config.")
        return None
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        if not torch.cuda.is_available():
            logger.warning("No GPU available; skipping LoRA fine-tune.")
            return None

        logger.info("Loading FinBERT for LoRA...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_CLS")
        model = get_peft_model(model, peft_config)

        # Tiny dataset for demonstration only (placeholder)
        ds = load_dataset("financial_phrasebank", "sentences_allagree")

        def tokenize(batch):
            return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

        tokenized = ds.map(tokenize, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        args = TrainingArguments(
            output_dir="artifacts/lora",
            per_device_train_batch_size=8,
            num_train_epochs=1,
            learning_rate=1e-4,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"])
        logger.info("Starting LoRA fine-tune (very small)...")
        trainer.train()
        save_path = "artifacts/lora/adapter"
        model.save_pretrained(save_path)
        logger.info(f"Saved LoRA adapter to {save_path}")
        return save_path
    except Exception as e:
        logger.warning(f"LoRA fine-tune skipped due to environment issue: {e}")
        return None




