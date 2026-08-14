import math

import torch
from omegaconf import DictConfig
from torch import nn
from transformers import GPT2LMHeadModel

from components.layers.attention import MultiHeadAttentionBlock
from components.layers.decoder import DecoderBlock, PositionwiseFeedForward


class GPT2(nn.Module):
    def __init__(
        self, model_configs: DictConfig, vocab_size: int, init_weights: bool = True
    ):
        super().__init__()
        self.model_configs = model_configs

        self.n_layers = model_configs.decoder.n_layers

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab_size, model_configs.decoder.embed_dim),
                "wpe": nn.Embedding(
                    model_configs.decoder.block_size, model_configs.decoder.embed_dim
                ),
                "drop": nn.Dropout(model_configs.decoder.dropout),
                "h": nn.ModuleList(
                    [
                        DecoderBlock(
                            embed_dim=model_configs.decoder.embed_dim,
                            self_attn=MultiHeadAttentionBlock(
                                embed_dim=model_configs.decoder.embed_dim,
                                n_heads=model_configs.decoder.n_heads,
                                dropout=model_configs.decoder.dropout,
                                cross_attn=False,
                                bias=model_configs.decoder.bias,
                            ),
                            cross_attention=MultiHeadAttentionBlock(
                                embed_dim=model_configs.decoder.embed_dim,
                                n_heads=model_configs.decoder.n_heads,
                                dropout=model_configs.decoder.dropout,
                                cross_attn=True,
                                bias=model_configs.decoder.bias,
                            ),
                            feed_forward=PositionwiseFeedForward(
                                embed_dim=model_configs.decoder.embed_dim,
                                d_ff=model_configs.decoder.feedforward_dim,
                                bias=model_configs.decoder.bias,
                                dropout=model_configs.decoder.dropout,
                            ),
                        )
                        for _ in range(model_configs.decoder.n_layers)
                    ]
                ),
                "ln_f": nn.LayerNorm(
                    model_configs.decoder.embed_dim, bias=model_configs.decoder.bias
                ),
            }
        )

        self.lm_head = nn.Linear(
            model_configs.decoder.embed_dim, vocab_size, bias=False
        )

        # NOTE: The idea is that similar tokens should have a similar embedding, (at token embedding), than you also expect that the probabilities
        # of the next token should be similar. As so, the weights of the last lm head layer are shared with the token embedding layer.
        # Also, the training process improves, since we do not have to train longer
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        if init_weights:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
                    )

        if model_configs.decoder.init_from_pretrained:
            self._load_from_gpt2()

        if model_configs.decoder.freeze_pretrained_layers:
            # freeze all layers except the cross-attention layers
            for name, param in self.named_parameters():
                if "cross_attention" not in name:
                    param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        encoder_memory: torch.Tensor,
        decoder_input_mask: torch.Tensor,
        cross_attn_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert idx.device == encoder_memory.device, (
            "Decoder input and image tokens must be on the same device"
        )

        device = idx.device
        _, t = idx.size()
        assert t <= self.model_configs.decoder.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.model_configs.block_size}"
        )

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(
            torch.arange(idx.size(1), dtype=torch.long, device=device)
        )
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x, encoder_memory, decoder_input_mask, cross_attn_masks)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _load_from_gpt2(self) -> None:
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_state_dict = gpt2_model.state_dict()
        transposable_weights = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        vocab_size_extenders = ["wte.weight", "lm_head.weight"]
        block_size_cappers = ["wpe.weight"]
        sd = self.state_dict()

        for k in gpt2_state_dict:
            if any(k.endswith(w) for w in transposable_weights):
                # special treatment for the Conv1D weights we need to transpose
                assert gpt2_state_dict[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(gpt2_state_dict[k].t())

            elif any(k.endswith(w) for w in vocab_size_extenders):
                # special treatment for the vocab size extenders
                assert gpt2_state_dict[k].shape[1] == sd[k].shape[1]
                with torch.no_grad():
                    sd[k][: gpt2_state_dict[k].shape[0], :].copy_(gpt2_state_dict[k])

            elif any(k.endswith(w) for w in block_size_cappers):
                # special treatment for the block size cappers
                assert gpt2_state_dict[k].shape[0] >= sd[k].shape[0]
                with torch.no_grad():
                    sd[k].copy_(gpt2_state_dict[k][: sd[k].shape[0], :])
            else:
                # vanilla copy over the other parameters
                assert gpt2_state_dict[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(gpt2_state_dict[k])
