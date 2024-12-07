import numpy as np
import torch

from transformers import ViTImageProcessor

from tokenizer import tokenizer


image_processor = ViTImageProcessor(image_mean=0.5, image_std=0.5)


def create_batch_processor(model):
    def process(batch):
        outputs = tokenizer(
                batch["grascii_normalized"],
                padding="max_length",
                max_length=12,
        )
        batch["labels"] = torch.tensor([[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in outputs.input_ids])
        batch["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
        batch["image"] = [np.array([i.convert(mode="L")]) for i in batch["image"]]
        batch["pixel_values"] = image_processor(batch["image"], return_tensors="pt").pixel_values
        return batch

    return process
