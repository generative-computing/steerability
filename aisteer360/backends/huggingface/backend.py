"""The in-process Hugging Face backend and its capability advertisement."""
from collections.abc import Callable

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.contracts import BackendCapabilities, Capability, CaptureKinds
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.backends.huggingface.session import ExclusiveSession
from aisteer360.utils.tokenization import ensure_pad_token

HF_CAPABILITIES = BackendCapabilities(
    atoms=frozenset({
        Capability.IN_PROCESS_TORCH,
        Capability.HIDDEN_CAPTURE,
        Capability.BEAM_PROPOSALS,
    }),
    capture_kinds=CaptureKinds(
        kinds=frozenset({"residual"}),
        locations=frozenset({"layer_output", "layer_input"}),
        modes=frozenset({"all_tokens", "last_token"}),
    ),
)


class HFBackend(Backend):
    """The in-process Hugging Face backend.

    Owns a loaded model and tokenizer, either loaded from a spec or adopted from a caller that
    already holds them. At most one session may be open per backend at a time, so the backend
    runs one generation at a time.
    """

    def __init__(
        self,
        spec: BackendSpec,
        *,
        model_provider: Callable[[], PreTrainedModel | None] | None = None,
        tokenizer_provider: Callable[[], object | None] | None = None,
    ) -> None:
        """Construct the backend, loading the model from `spec` unless providers are given.

        Loading reads the options `hf_model_kwargs`, `device_map`, `tokenizer_name_or_path`,
        and `trust_remote_code`. Option values must be plain data, since spec canonicalization
        renders live objects (e.g. a quantization config instance) as strings that
        `from_pretrained` cannot consume. A `device_map` key inside `hf_model_kwargs` is used
        when the spec carries no top-level `device_map` option.

        Args:
            spec: The backend spec.
            model_provider: Callable returning the adopted model; used with
                `tokenizer_provider` instead of loading.
            tokenizer_provider: Callable returning the adopted tokenizer.

        Raises:
            ValueError: If `spec.kind` is not `"huggingface"`, or no model reference is
                available to load from.
        """
        if spec.kind != "huggingface":
            raise ValueError(f"HFBackend requires a 'huggingface' spec; got kind {spec.kind!r}.")
        self.spec = spec
        self._open_session: ExclusiveSession | None = None

        if model_provider is not None:
            self._model_provider = model_provider
            self._tokenizer_provider = tokenizer_provider or (lambda: None)
            return

        if spec.model is None:
            raise ValueError(
                "HFBackend needs a model reference on the spec, or model_provider/"
                "tokenizer_provider for an already-loaded model."
            )
        hf_model_kwargs = dict(spec.get_option("hf_model_kwargs", default={}))
        device_map = spec.get_option("device_map", default=hf_model_kwargs.pop("device_map", "auto"))
        model = AutoModelForCausalLM.from_pretrained(
            spec.model,
            device_map=device_map,
            **hf_model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.get_option("tokenizer_name_or_path") or spec.model,
            trust_remote_code=bool(spec.get_option("trust_remote_code", default=False)),
        )
        tokenizer = ensure_pad_token(tokenizer)
        self._model_provider = lambda: model
        self._tokenizer_provider = lambda: tokenizer

    @classmethod
    def adopt(
        cls,
        spec: BackendSpec,
        model_provider: Callable[[], PreTrainedModel | None],
        tokenizer_provider: Callable[[], object | None],
    ) -> "HFBackend":
        """Wrap an already-loaded model and tokenizer without loading anything.

        Providers are read on every access, so a caller whose model is replaced mid-steer (a
        structural control returning a new model) always exposes the current one to sessions.

        Args:
            spec: The backend spec identifying this configuration.
            model_provider: Callable returning the current model (may return None before one
                exists).
            tokenizer_provider: Callable returning the current tokenizer.

        Returns:
            The adopting backend.
        """
        return cls(spec, model_provider=model_provider, tokenizer_provider=tokenizer_provider)

    @classmethod
    def capabilities_for_spec(cls, spec: BackendSpec) -> BackendCapabilities:
        """The static Hugging Face capability advertisement (spec-independent)."""
        return HF_CAPABILITIES

    def open_session(self) -> "ExclusiveSession":
        """Open the backend's one exclusive session.

        Returns:
            The session, usable as a context manager.

        Raises:
            RuntimeError: If an exclusive session is already open on this backend.
        """
        if self._open_session is not None and not self._open_session.closed:
            raise RuntimeError(
                "An exclusive session is already open on this backend; close it before opening "
                "another."
            )
        self._open_session = ExclusiveSession(self)
        return self._open_session
