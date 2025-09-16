"""LoRA fine-tuning for FinBERT on financial sentiment data.

This module provides realistic LoRA fine-tuning that demonstrates production-level
practices including proper evaluation, logging, and hyperparameter tuning.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.utils.logging import get_logger


logger = get_logger(__name__)


def maybe_lora_finetune(enabled: bool, model_id: str = "yiyanghkust/finbert-tone") -> Optional[str]:
    """Run LoRA fine-tuning with proper evaluation and logging.

    Returns the path to a saved adapter, or None if skipped.
    """
    if not enabled:
        logger.info("LoRA fine-tune disabled via config.")
        return None
    
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoModelForSequenceClassification, 
            AutoTokenizer, 
            Trainer, 
            TrainingArguments,
            EvalPrediction
        )

        if not torch.cuda.is_available():
            logger.warning("No GPU available; skipping LoRA fine-tune.")
            return None

        logger.info("Starting LoRA fine-tuning of FinBERT...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        # More LoRA configuration
        peft_config = LoraConfig(
            r=16,                           # Higher rank for better adaptation
            lora_alpha=32,                  # Higher alpha for stronger adaptation
            lora_dropout=0.1,               # Standard dropout
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", "value", "key", "dense"]  # Target more modules
        )
        
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
        
        # Load and prepare dataset
        logger.info("Loading Financial PhraseBank dataset...")
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["sentence"], 
                truncation=True, 
                padding="max_length", 
                max_length=256,  # Longer sequences for financial text
                return_tensors="pt"
            )
        
        # Tokenize datasets
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Split train into train/val for proper evaluation
        train_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2, seed=42)
        
        # Evaluation function
        def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted"),
            }
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="artifacts/lora",
            num_train_epochs=3,                    
            per_device_train_batch_size=16,       
            per_device_eval_batch_size=32,
            warmup_steps=100,                     
            weight_decay=0.01,                     
            logging_dir="artifacts/lora/logs",
            logging_steps=50,
            eval_steps=100,                       
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=[],                        
            gradient_checkpointing=True,           
            dataloader_pin_memory=True,
            learning_rate=2e-4,                    
            lr_scheduler_type="cosine",     # Cosine annealing
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["test"],
            compute_metrics=compute_metrics,
        )
        
        # Get baseline metrics before fine-tuning
        logger.info("Evaluating baseline model...")
        baseline_metrics = trainer.evaluate()
        baseline_metrics = {f"baseline_{k}": v for k, v in baseline_metrics.items()}
        
        # Start MLflow run for LoRA fine-tuning
        with mlflow.start_run(run_name="lora_finetune"):
            # Log configuration
            mlflow.log_params({
                "model_id": model_id,
                "lora_r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
                "target_modules": str(peft_config.target_modules),
                "learning_rate": training_args.learning_rate,
                "num_epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "max_length": 256,
            })
            
            # Log baseline metrics
            for key, value in baseline_metrics.items():
                mlflow.log_metric(key, value)
            
            # Fine-tune
            logger.info("Starting LoRA fine-tuning...")
            train_result = trainer.train()
            
            # Final evaluation
            logger.info("Evaluating fine-tuned model...")
            final_metrics = trainer.evaluate()
            
            # Log training and final metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "final_accuracy": final_metrics["eval_accuracy"],
                "final_f1": final_metrics["eval_f1"],
                "improvement_accuracy": final_metrics["eval_accuracy"] - baseline_metrics["baseline_eval_accuracy"],
                "improvement_f1": final_metrics["eval_f1"] - baseline_metrics["baseline_eval_f1"],
            })
            
            # Save adapter
            save_path = "artifacts/lora/adapter"
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Log adapter as MLflow artifact
            mlflow.log_artifacts(save_path, "lora_adapter")
            
            logger.info(f"LoRA fine-tuning completed!")
            logger.info(f"Baseline accuracy: {baseline_metrics['baseline_eval_accuracy']:.4f}")
            logger.info(f"Fine-tuned accuracy: {final_metrics['eval_accuracy']:.4f}")
            logger.info(f"Improvement: {final_metrics['eval_accuracy'] - baseline_metrics['baseline_eval_accuracy']:.4f}")
            logger.info(f"Saved adapter to {save_path}")
            
            return save_path
            
    except Exception as e:
        logger.warning(f"LoRA fine-tune failed: {e}")
        return None




