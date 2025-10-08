import torch
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel

class MultiTaskTrOCR(VisionEncoderDecoderModel):
    """
    Skeleton for a TrOCR-based multi-task model.
    Add a token-wise classification head over decoder hidden states to predict per-character font group.
    """
    def __init__(self, config, num_fonts: int = 8, font_pad_token_id: int = -100, lambda_font: float = 1.0):
        super().__init__(config)
        self.font_head = torch.nn.Linear(self.config.decoder.hidden_size, num_fonts)
        self.font_pad_token_id = font_pad_token_id
        self.lambda_font = lambda_font

    def forward(self, pixel_values=None, labels=None, font_labels=None, **kwargs):
        out = super().forward(pixel_values=pixel_values, labels=labels, output_hidden_states=True, **kwargs)
        loss = out.loss
        font_logits = None
        if font_labels is not None and out.decoder_hidden_states is not None:
            dec_h = out.decoder_hidden_states[-1]  # [B, T, H]
            font_logits = self.font_head(dec_h)    # [B, T, C]
            loss_font = F.cross_entropy(
                font_logits.reshape(-1, font_logits.size(-1)),
                font_labels.reshape(-1),
                ignore_index=self.font_pad_token_id,
            )
            loss = loss + self.lambda_font * loss_font
        out.loss = loss
        out.font_logits = font_logits
        return out
