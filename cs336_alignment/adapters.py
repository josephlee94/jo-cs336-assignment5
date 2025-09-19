# cs336_alignment/adapters.py
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    # tokenize separately
    tok_prompts = tokenizer(prompt_strs, padding=True, truncation=True, return_tensors="pt")
    tok_outputs = tokenizer(output_strs, padding=True, truncation=True, return_tensors="pt")

    # concat (drop last token of prompt to avoid double BOS, optional)
    input_ids = []
    labels = []
    response_mask = []

    for i in range(len(prompt_strs)):
        p = tok_prompts.input_ids[i]
        o = tok_outputs.input_ids[i]

        # concat with no extra special tokens
        concat = torch.cat([p, o], dim=0)

        # Shifted labels: predict token t from tokens < t
        # Labels align with concat[1:], inputs with concat[:-1]
        input_ids.append(concat[:-1])
        labels.append(concat[1:])

        # mask = 1 for response (the part coming from output), 0 for prompt
        rm = torch.zeros_like(labels[-1])
        rm[len(p)-1:] = 1   # -1 aligns because of the shift
        response_mask.append(rm)

    # pad to same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 ignored by CE
    response_mask = torch.nn.utils.rnn.pad_sequence(response_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask.bool(),
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=-1)            # (B, T)
    return ent

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        logits = model(input_ids).logits  # (B, T, V)

    # gather log p(y_t | x, y_<t>)
    log_probs = F.log_softmax(logits, dim=-1)
    B, T = labels.shape
    # labels may contain -100 for padding -> clamp for gather, then mask later
    gather_labels = labels.clamp(min=0)
    token_logp = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    token_logp = token_logp.masked_fill(labels.eq(-100), 0.0)

    out = {"log_probs": token_logp}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float, dim: Optional[int] = None) -> torch.Tensor:
    # sum over dim with mask, divide by constant (avoid div0)
    masked = tensor * mask.to(tensor.dtype)
    if dim is None:
        s = masked.sum()
    else:
        s = masked.sum(dim=dim)
    return s / max(normalize_constant, 1e-8)
