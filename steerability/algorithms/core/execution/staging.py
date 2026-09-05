"""Stage mechanics for the engine-backed steer phase: the stage free protocol and the
steer-time capture smoke test.

The free protocol takes only a `weakref.ref`; a strong reference crossing the function
boundary would keep the staged model alive through `gc.collect()` and defeat the check.
"""
from __future__ import annotations

import gc
import weakref

import torch

from steerability.algorithms.core.execution.payloads import PreparedPrompt


def capture_smoke_failure(session, fallback_tokenizer=None) -> str | None:
    """Issue one single-prompt capture through `session`; the error text on failure.

    The probe token id comes from the session's tokenizer, else `fallback_tokenizer`.
    """
    tokenizer = getattr(session, "tokenizer", None) or fallback_tokenizer
    token_id = 0
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            token_id = int(value)
            break
    prompt = PreparedPrompt.from_token_ids(torch.tensor([[token_id]], dtype=torch.long))
    try:
        session.capture([prompt], layers=[0], mode="last_token", location="layer_output")
    except Exception as error:
        return str(error)
    return None


def verify_stage_released(ref: weakref.ref, controls) -> None:
    """Verify the staged in-process model's weights are actually gone.

    Runs a collection pass, clears the CUDA cache, and dereferences `ref` (the caller must
    have dropped its own strong reference first).

    Raises:
        RuntimeError: If a control retained the staged model past the stage; the message
            names the retaining controls where identifiable.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    referent = ref()
    if referent is None:
        return
    holders = find_model_holders(referent, controls)
    names = ", ".join(holders) if holders else "an unidentified holder"
    raise RuntimeError(
        f"The staged in-process model was retained past the steer stage by: {names}. "
        "Controls supported at generate on this backend must not hold the pipeline model "
        "beyond steer(); release every instance reference to it before steer() returns, or "
        "require Capability.IN_PROCESS_TORCH at generate."
    )


def find_model_holders(referent, controls) -> list[str]:
    """Controls holding `referent` in their instance attributes (one level) or in a bound
    intervention's transform or gate attributes."""

    def instance_values(obj):
        try:
            return list(vars(obj).values())
        except TypeError:
            slots = getattr(type(obj), "__slots__", ())
            return [getattr(obj, name, None) for name in slots]

    holders: list[str] = []
    for control in controls:
        found = any(value is referent for value in instance_values(control))
        if not found:
            for intervention in getattr(control, "interventions", ()) or ():
                for slot in (
                    getattr(intervention, "transform", None),
                    getattr(intervention, "gate", None),
                ):
                    if slot is not None and any(
                        value is referent for value in instance_values(slot)
                    ):
                        found = True
        if found:
            holders.append(type(control).__name__)
    return holders
