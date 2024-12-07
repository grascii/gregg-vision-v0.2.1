from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from metrics import compute_metrics
from model import create_model
from preprocess import create_batch_processor
from tokenizer import tokenizer


dataset = load_dataset(
    "grascii/gregg-preanniversary-words",
    split="train",
    revision="0227610b8aa2cd5587fe6c247b355746825b8b3c",
)


model = create_model()

preprocess_batch = create_batch_processor(model)

dataset = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=128,
    remove_columns=["image", "grascii_normalized", "longhand"]
)

dataset = dataset.train_test_split(test_size=0.2, train_size=0.8, seed=42)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    output_dir="./training",
    logging_steps=2,
    save_steps=33,
    num_train_epochs=8,
    eval_steps=33,
    save_total_limit=5,
    fp16=True,
    learning_rate=1e-4,
    generation_max_length=12,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-6},
    warmup_steps=99,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"].select(range(256)),
)

trainer.train()

model.save_pretrained("gregg-vision-v0.2.1")
