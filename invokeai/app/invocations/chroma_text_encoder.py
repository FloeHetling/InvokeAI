from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from transformers import PreTrainedTokenizerBase, T5EncoderModel

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import T5EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ChromaConditioningInfo, ConditioningFieldData
from invokeai.backend.util.fp8 import get_model_compute_dtype


@dataclass(frozen=True)
class _ChromaPromptTokenization:
    input_ids: torch.Tensor
    token_weights: torch.Tensor

    @property
    def has_weights(self) -> bool:
        return bool(torch.any(self.token_weights != 1.0).item())


@contextmanager
def _t5_layerwise_fp32_compute(text_encoder: T5EncoderModel, storage_dtype: torch.dtype) -> Iterator[None]:
    """Run T5 blocks with FP32 activations/weights while keeping the cached model in its storage dtype.

    Raw FP8 T5 checkpoints are materialized in BF16 for safe execution. Because every e4m3fn value
    is exactly representable in BF16, temporarily upcasting one block at a time recreates the original
    FP8 values in FP32 without requiring a full FP32 T5 copy.
    """
    blocks = list(text_encoder.encoder.block)
    final_layer_norm = text_encoder.encoder.final_layer_norm
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def pre_forward(module: torch.nn.Module, args: tuple[object, ...]) -> tuple[object, ...] | None:
        module.to(dtype=torch.float32)
        if not args or not isinstance(args[0], torch.Tensor):
            return None
        return (args[0].to(dtype=torch.float32), *args[1:])

    def post_forward(module: torch.nn.Module, args: tuple[object, ...], output: object) -> object:
        module.to(dtype=storage_dtype)
        return output

    try:
        for block in blocks:
            handles.append(block.register_forward_pre_hook(pre_forward))
            handles.append(block.register_forward_hook(post_forward, always_call=True))
        final_layer_norm.to(dtype=torch.float32)
        yield
    finally:
        for handle in handles:
            handle.remove()
        # The always-call hooks restore blocks on normal/error exits from each forward. This final
        # pass also covers an exception raised by a pre-hook before the matching forward hook runs.
        for block in blocks:
            block.to(dtype=storage_dtype)
        final_layer_norm.to(dtype=storage_dtype)


@invocation(
    "chroma_text_encoder",
    title="Prompt - Chroma",
    tags=["prompt", "conditioning", "chroma", "t5"],
    category="prompt",
    version="1.0.0",
    idle_gpu_offloadable=True,
)
class ChromaTextEncoderInvocation(BaseInvocation):
    """Encode a Chroma prompt with variable-length T5-XXL conditioning."""

    t5_encoder: T5EncoderField = InputField(
        title="T5 Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )

    @classmethod
    def _parse_prompt_attention(cls, prompt: str, weight: float = 1.0) -> list[tuple[str, float]]:
        """Parse parenthesized prompt weights into independently tokenized segments.

        Parenthesized text gets a 1.1x weight per nesting level. A trailing numeric
        suffix, e.g. ``(detail:1.35)``, sets that group's weight explicitly. Escaped
        parentheses are treated as literal text. Segment boundaries are preserved so
        each weighted span can be tokenized independently before concatenation.
        """

        def find_closing_paren(text: str, opening_index: int) -> int | None:
            depth = 1
            index = opening_index + 1
            while index < len(text):
                if text[index] == "\\" and index + 1 < len(text) and text[index + 1] in "()":
                    index += 2
                    continue
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        return index
                index += 1
            return None

        def unescape_parentheses(text: str) -> str:
            return text.replace("\\(", "(").replace("\\)", ")")

        segments: list[tuple[str, float]] = []
        cursor = 0
        literal_start = 0
        while cursor < len(prompt):
            if prompt[cursor] == "\\" and cursor + 1 < len(prompt) and prompt[cursor + 1] in "()":
                cursor += 2
                continue
            if prompt[cursor] != "(":
                cursor += 1
                continue

            closing_index = find_closing_paren(prompt, cursor)
            if closing_index is None:
                if cursor > literal_start:
                    segments.append((unescape_parentheses(prompt[literal_start:cursor]), weight))
                segments.append((unescape_parentheses(prompt[cursor:]), weight))
                return segments

            if cursor > literal_start:
                segments.append((unescape_parentheses(prompt[literal_start:cursor]), weight))

            group_text = prompt[cursor + 1 : closing_index]
            group_weight = weight * 1.1
            weight_separator = group_text.rfind(":")
            if weight_separator > 0:
                try:
                    group_weight = float(group_text[weight_separator + 1 :])
                    group_text = group_text[:weight_separator]
                except ValueError:
                    pass

            segments.extend(cls._parse_prompt_attention(group_text, group_weight))
            cursor = closing_index + 1
            literal_start = cursor

        if literal_start < len(prompt):
            segments.append((unescape_parentheses(prompt[literal_start:]), weight))
        return segments

    @classmethod
    def tokenize_prompt(cls, tokenizer: PreTrainedTokenizerBase, prompt: str) -> _ChromaPromptTokenization:
        """Tokenize variable-length weighted Chroma text without fixed-length padding.

        Weighted segments are tokenized independently, each segment's EOS is removed,
        then the segments are concatenated and one final EOS is appended. The sequence
        is not truncated to 512 tokens and no minimum padding is added.
        """
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("The selected Chroma T5 tokenizer has no EOS token")

        token_ids: list[int] = []
        token_weights: list[float] = []
        for text, weight in cls._parse_prompt_attention(prompt):
            if text == "":
                continue
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            segment_ids = list(encoded["input_ids"])
            if segment_ids and segment_ids[-1] == eos_token_id:
                segment_ids.pop()
            token_ids.extend(segment_ids)
            token_weights.extend([weight] * len(segment_ids))

        token_ids.append(eos_token_id)
        token_weights.append(1.0)
        return _ChromaPromptTokenization(
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            token_weights=torch.tensor([token_weights], dtype=torch.float32),
        )

    @staticmethod
    def apply_prompt_weights(
        prompt_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        token_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply embedding-space prompt weighting relative to a neutral baseline in float32."""
        weights = token_weights.to(device=prompt_embeds.device, dtype=torch.float32).unsqueeze(-1)
        prompt_float = prompt_embeds.float()
        baseline_float = baseline_embeds.float()
        return (prompt_float - baseline_float) * weights + baseline_float

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        text_encoder_info = context.models.load(self.t5_encoder.text_encoder)
        text_encoder_config = text_encoder_info.config
        if text_encoder_config is None:
            raise ValueError("The selected T5 encoder has no model configuration")

        with (
            text_encoder_info.model_on_device() as (cached_weights, text_encoder),
            context.models.load(self.t5_encoder.tokenizer) as tokenizer,
            ExitStack() as exit_stack,
        ):
            if not isinstance(text_encoder, T5EncoderModel):
                raise TypeError(f"Expected T5EncoderModel, got {type(text_encoder).__name__}")
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"Expected a Transformers tokenizer, got {type(tokenizer).__name__}")

            compute_dtype = get_model_compute_dtype(text_encoder)
            raw_checkpoint = text_encoder_config.format is ModelFormat.Checkpoint
            # Moving patched blocks between BF16 and FP32 would invalidate active LoRA sidecars.
            t5_fp32_compute = raw_checkpoint and not self.t5_encoder.loras
            force_sidecar = text_encoder_config.format not in {ModelFormat.T5Encoder, ModelFormat.Diffusers}
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._t5_lora_iterator(context),
                    prefix=FLUX_LORA_T5_PREFIX,
                    dtype=compute_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=force_sidecar,
                )
            )

            tokenized = self.tokenize_prompt(tokenizer, self.prompt)
            device = text_encoder_info.compute_device
            input_ids = tokenized.input_ids.to(device)

            if t5_fp32_compute:
                context.logger.info(
                    f"Chroma layerwise FP32 T5 compute enabled "
                    f"(storage_dtype={compute_dtype}, blocks={len(text_encoder.encoder.block)})."
                )

            if context.config.get().log_tokenization:
                context.logger.info(
                    f">> [CHROMA T5 TOKENLOG] Tokens ({input_ids.shape[1]}, variable-length; "
                    f"weighted={'yes' if tokenized.has_weights else 'no'})"
                )

            # The prompt sequence itself is unpadded, so every prompt token participates in
            # T5 self-attention and no attention mask is required. For weighted prompts,
            # encode a same-length all-PAD baseline in the same pass and use it as the
            # neutral reference for embedding-space weighting.
            encoder_input_ids = input_ids
            if tokenized.has_weights:
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    raise ValueError("The selected Chroma T5 tokenizer has no PAD token")
                baseline_input_ids = torch.full_like(input_ids, pad_token_id)
                encoder_input_ids = torch.cat((input_ids, baseline_input_ids), dim=0)

            context.util.signal_progress("Running Chroma T5 encoder")
            compute_context = (
                _t5_layerwise_fp32_compute(text_encoder, compute_dtype) if t5_fp32_compute else nullcontext()
            )
            with compute_context:
                encoded = text_encoder(
                    encoder_input_ids,
                    attention_mask=None,
                    output_hidden_states=False,
                    return_dict=False,
                )[0]
            prompt_embeds = encoded[:1]

            if tokenized.has_weights:
                prompt_embeds = self.apply_prompt_weights(
                    prompt_embeds=prompt_embeds,
                    baseline_embeds=encoded[1:2],
                    token_weights=tokenized.token_weights,
                )

            conditioning_dtype = torch.float32 if t5_fp32_compute else compute_dtype
            prompt_embeds = prompt_embeds.to(dtype=conditioning_dtype)
            # The stored sequence itself contains no padding. The Chroma adapter only
            # introduces masked right-padding later when positive and negative branches
            # of different lengths are combined into one optimized CFG batch.
            prompt_attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.bool, device=prompt_embeds.device)

        conditioning_data = ConditioningFieldData(
            conditionings=[
                ChromaConditioningInfo(
                    prompt_embeds=prompt_embeds.detach().to("cpu"),
                    prompt_attention_mask=prompt_attention_mask.detach().to("cpu"),
                )
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _t5_lora_iterator(self, context: InvocationContext) -> Iterator[PatchSpec]:
        for lora in self.t5_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(f"Expected ModelPatchRaw LoRA, got {type(lora_info.model).__name__}")
            yield (lora_info.model, lora.weight, lora_info.model_in_ram())
