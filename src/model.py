from transformers import VisionEncoderDecoderModel, RobertaConfig, RobertaModel

from tokenizer import tokenizer


def create_model():
    decoder_config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=6,
        num_attention_heads=6,
        hidden_size=384,
        intermediate_size=1536,
        max_position_embeddings=32,
        type_vocab_size=1,
        is_decoder=True,
    )
    RobertaModel(decoder_config).save_pretrained("decoder")

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "grascii/vit-small-patch16-224-single-channel", "decoder"
    )

    model.config.decoder_start_token_id = 0
    model.config.bos_token_id = 0
    model.config.pad_token_id = 1
    model.config.eos_token_id = 2
    model.generation_config.decoder_start_token_id = 0

    return model
