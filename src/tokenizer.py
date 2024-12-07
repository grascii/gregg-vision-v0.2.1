from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "src/tokenizer",
    model_input_names=["labels", "pixel_values"],
)
