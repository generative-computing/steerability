"""`SPipe`: a portable serialization of a `SteeringPipeline`.

One format holds both the recipe and the frozen resolution. A recipe-only spipe carries the
model reference and the controls as constructed; loading it and calling `steer()` re-runs
fits. A frozen spipe additionally pins what the resolution produced (fingerprints, resolved
bindings, per-fit digests) in a lock section and stores the products content-addressed. The
loaded controls therefore steer cheaply and model-free. The frozen form is itself a valid
recipe, since every resolved entry is constructor-valid for its method and loading takes the
ordinary construction and `steer()` path.
"""
from __future__ import annotations

import copy
import logging
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from steerability.spipe.codec import DecodeContext, decode, digest_of
from steerability.spipe.errors import SpipeFormatError, SpipeIntegrityError, SpipeSaveError, SpipeStaleError
from steerability.spipe.format import (
    ARTIFACTS_DIR,
    MANIFEST_NAME,
    pack_zip,
    read_manifest,
    unpack_zip,
    validate_manifest,
    write_manifest,
)
from steerability.spipe.store import ArtifactRecord, ArtifactStore

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.spec import BackendSpec
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

logger = logging.getLogger(__name__)


@dataclass
class SpipeReport:
    """The result of a model-free `SPipe.verify()`.

    Attributes:
        ok: True when no errors were found.
        errors: Findings that make the bundle unusable as-is.
        warnings: Findings worth knowing that do not block loading.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        """A human-readable summary."""
        lines = [f"spipe verify: {'ok' if self.ok else 'FAILED'}"]
        lines.extend(f"  error: {message}" for message in self.errors)
        lines.extend(f"  warning: {message}" for message in self.warnings)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _resolved_items(entry: Mapping) -> list[Mapping]:
    resolved = entry.get("resolved")
    if resolved is None:
        return []
    return list(resolved) if isinstance(resolved, list) else [resolved]


def _collect_artifact_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "$artifact" and isinstance(item, str):
                found.add(item)
            elif key == "id" and isinstance(item, str) and item.startswith("sha256:"):
                found.add(item)
            else:
                _collect_artifact_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_artifact_ids(item, found)


def _manifest_records(manifest: Mapping) -> dict[str, ArtifactRecord]:
    """The artifact records of every resolved entry, keyed by artifact id."""
    records: dict[str, ArtifactRecord] = {}
    for entry in manifest["controls"]:
        for item in _resolved_items(entry):
            for record in (item.get("artifacts") or {}).values():
                records[record["id"]] = ArtifactRecord.from_mapping(record)
    return records


class SPipe:
    """A serialized steering pipeline: recipe, optional lock, and artifact store.

    Instances come from `SteeringPipeline.to_spipe()` (backed by a temporary store until
    saved) or `SPipe.load()` (backed by the loaded bundle). The manifest is data; controls
    are instantiated only by `pipeline()`.
    """

    def __init__(
        self,
        manifest: dict,
        *,
        store: ArtifactStore | None,
        base_dir: Path | None,
        allow_code: bool,
        allow_stale: bool = False,
        _temp: Any = None,
    ):
        validate_manifest(manifest)
        self._manifest = manifest
        self._manifest_records = _manifest_records(manifest)
        self._store = store
        self._base_dir = base_dir
        self._allow_code = allow_code
        self._allow_stale = allow_stale
        self._temp = _temp
        self._staleness_checked = False

    # inspection

    @property
    def manifest(self) -> dict:
        """A deep copy of the manifest."""
        return copy.deepcopy(self._manifest)

    @property
    def recipe_id(self) -> str:
        """The recipe identity digest (model reference plus controls), 12 hex characters."""
        lock = self._manifest.get("lock")
        if lock is not None:
            return lock["recipe_id"]
        from steerability.algorithms.core.identity import config_digest

        return config_digest({
            "model": self._manifest["model"]["ref"],
            "controls": self._descriptor()["controls"],
        })

    @property
    def config_id(self) -> str:
        """The configuration identity digest over the recipe entries, 12 hex characters."""
        lock = self._manifest.get("lock")
        if lock is not None:
            return lock["config_id"]
        from steerability.algorithms.core.identity import config_digest

        return config_digest(self._descriptor())

    def _descriptor(self) -> dict:
        """The configuration descriptor recomputed from freshly instantiated recipe controls."""
        from steerability.algorithms.core.identity import config_descriptor_from_controls

        controls = [self._instantiate(entry["method"], entry["args"], self._decode_ctx(lenient=True))
                    for entry in self._manifest["controls"]]
        for control, entry in zip(controls, self._manifest["controls"]):
            control.enabled = entry["enabled"]
        return config_descriptor_from_controls(controls)

    @property
    def code_dependent(self) -> bool:
        """Whether loading this spipe needs matching code and `allow_code=True`."""
        return bool(self._manifest["code_dependent"])

    @property
    def is_frozen(self) -> bool:
        """True iff every enabled entry with steer-time products carries a resolution.

        Entries with `resolved: null` have nothing to pin (their recipe is their frozen
        form), which makes a manifest with a lock section frozen.
        """
        return self._manifest.get("lock") is not None

    def describe(self) -> str:
        """A human-readable table of entries, frozen state, and artifact sizes."""
        lines = [
            f"spipe {self._manifest['format']}  model={self._manifest['model']['ref']}",
            f"  recipe_id={self.recipe_id}  config_id={self.config_id}  "
            f"frozen={self.is_frozen}  code_dependent={self.code_dependent}",
        ]
        for i, entry in enumerate(self._manifest["controls"]):
            items = _resolved_items(entry)
            frozen = "recipe"
            if items:
                methods = ", ".join(item["method"] for item in items)
                frozen = f"frozen -> {methods}"
            elif self.is_frozen and entry["enabled"]:
                frozen = "frozen (recipe is the frozen form)"
            enabled = "" if entry["enabled"] else "  [disabled]"
            lines.append(f"  [{i}] {entry['method']}{enabled}  ({frozen})")
            for item in items:
                for name, record in (item.get("artifacts") or {}).items():
                    size = ""
                    if self._store is not None and self._store.has(record["id"]):
                        size = f"  {self._store.size_of(record['id'])} bytes"
                    lines.append(f"        {name}: {record['type']} {record['id'][:19]}…{size}")
        return "\n".join(lines)

    # freeze state

    def thaw(self) -> "SPipe":
        """A recipe-only copy: every `resolved` section and the lock are dropped."""
        manifest = self.manifest
        for entry in manifest["controls"]:
            entry["resolved"] = None
        manifest["lock"] = None
        return SPipe(
            manifest, store=self._store, base_dir=self._base_dir,
            allow_code=self._allow_code, allow_stale=self._allow_stale, _temp=self._temp,
        )

    # verification

    def _referenced_artifact_ids(self) -> set[str]:
        found: set[str] = set()
        _collect_artifact_ids(self._manifest["controls"], found)
        return found

    def verify(self) -> SpipeReport:
        """The model-free report: format, integrity, staleness, versions, code dependence.

        Never loads a model or instantiates a backend. Referenced artifacts that are not
        present are reported in one warning (a thin bundle); the present ones are verified
        against their content ids.

        Returns:
            The report.
        """
        errors: list[str] = []
        report_warnings: list[str] = []

        try:
            validate_manifest(self._manifest)
        except SpipeFormatError as exc:
            errors.append(str(exc))

        referenced = sorted(self._referenced_artifact_ids())
        missing = [
            artifact_id for artifact_id in referenced
            if self._store is None or not self._store.has(artifact_id)
        ]
        if missing:
            report_warnings.append(
                f"{len(missing)} of {len(referenced)} referenced artifact(s) are not present (thin "
                f"bundle); pass artifact_store= at load to resolve them: {', '.join(missing)}."
            )
        for artifact_id in referenced:
            if artifact_id not in missing:
                try:
                    self._store.verify(artifact_id)
                except SpipeIntegrityError as exc:
                    errors.append(str(exc))

        try:
            self._check_staleness(raise_on_stale=True, force=True)
        except SpipeStaleError as exc:
            errors.append(str(exc))
        except Exception as exc:
            report_warnings.append(f"staleness check could not run: {exc}")

        from steerability.spipe.freeze import _toolkit_version

        saved = self._manifest.get("toolkit_version", "unknown")
        current = _toolkit_version()
        if saved.split(".")[0] != current.split(".")[0]:
            report_warnings.append(
                f"spipe was written by steerability {saved}; this is {current} (major mismatch)."
            )
        if self.code_dependent:
            report_warnings.append(
                "the manifest references code ($ref); loading a pipeline from it requires "
                "allow_code=True and the referenced modules on the import path."
            )

        return SpipeReport(ok=not errors, errors=errors, warnings=report_warnings)

    def _decode_ctx(self, *, lenient: bool, verify: str = "off", data_mode: str = "keep") -> DecodeContext:
        return DecodeContext(
            store=self._store,
            allow_code=self._allow_code,
            code_mode="sentinel" if lenient else "strict",
            verify=verify,
            data_mode=data_mode,
            manifest_records=self._manifest_records,
        )

    @staticmethod
    def _instantiate(method_key: str, encoded_args: Mapping, ctx: DecodeContext):
        from steerability.algorithms.core.registry import RegistryError, resolve_method_key

        try:
            method = resolve_method_key(method_key)
        except RegistryError as exc:
            raise SpipeFormatError(str(exc)) from exc
        kwargs = {name: decode(value, ctx, f"args.{name}") for name, value in encoded_args.items()}
        return method.control_cls(**kwargs)

    def _check_staleness(self, *, raise_on_stale: bool, force: bool = False) -> None:
        """Recompute each frozen entry's fit digest from the current recipe args.

        Skipped when artifacts are unavailable (thin bundle without a store); `pipeline()`
        re-invokes it once artifacts resolve.

        Raises:
            SpipeStaleError: If a recorded fit digest does not match the recomputed one, or a
                recipe entry can no longer be reconstructed for the check.
        """
        if self._staleness_checked and not force:
            return
        for i, entry in enumerate(self._manifest["controls"]):
            recorded: dict[str, str] = {}
            for item in _resolved_items(entry):
                for name, record in (item.get("artifacts") or {}).items():
                    if record.get("fit_digest"):
                        recorded[name] = record["fit_digest"]
            if not recorded:
                continue
            try:
                control = self._instantiate(entry["method"], entry["args"], self._decode_ctx(lenient=True))
                fit_identity = control.fit_identity()
            except Exception as exc:
                raise SpipeStaleError(
                    f"controls[{i}] ({entry['method']}): the recipe args no longer reconstruct "
                    f"for the staleness check ({exc}); thaw() and re-steer(), or pass "
                    "allow_stale=True."
                ) from exc
            current = digest_of(fit_identity) if fit_identity is not None else None
            for name, digest in recorded.items():
                if digest != current:
                    message = (
                        f"controls[{i}] ({entry['method']}): frozen artifact {name!r} was "
                        f"produced from fit digest {digest} but the recipe now digests to "
                        f"{current}; the fit-relevant recipe fields were edited after "
                        "freezing. thaw() and re-steer(), or pass allow_stale=True."
                    )
                    if raise_on_stale:
                        raise SpipeStaleError(message)
                    warnings.warn(message, UserWarning)
        self._staleness_checked = True

    # load / save

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_code: bool = False,
        allow_stale: bool = False,
        artifact_store: str | Path | Callable[[str], Path] | None = None,
    ) -> "SPipe":
        """Load a `.spipe` file or directory.

        Args:
            path: A `.spipe` zip file or a spipe directory.
            allow_code: Permit `$ref` imports, non-`steerability.` `$dc` imports, and
                pickle-backed memory payloads during decoding.
            allow_stale: Skip the staleness check.
            artifact_store: External artifact source for thin bundles: a store directory, or
                a callable mapping an artifact id to the directory holding it.

        Returns:
            The loaded `SPipe`.

        Raises:
            SpipeFormatError: On format-version, schema, or archive violations.
            SpipeIntegrityError: If a present artifact fails content verification.
            SpipeStaleError: If a frozen entry is stale and `allow_stale` is False.
        """
        path = Path(path)
        temp = None
        if path.is_dir():
            base_dir = path
        else:
            temp = tempfile.TemporaryDirectory(prefix="spipe-load-")
            unpack_zip(path, temp.name)
            base_dir = Path(temp.name)

        manifest = read_manifest(base_dir)

        resolver: Callable[[str], Path] | None = None
        if callable(artifact_store):
            resolver = artifact_store
        elif artifact_store is not None:
            external = Path(artifact_store)
            resolver = lambda artifact_id: external / artifact_id.replace(":", "-", 1)  # noqa: E731
        store = ArtifactStore(base_dir / ARTIFACTS_DIR, resolver=resolver)

        spipe = cls(
            manifest, store=store, base_dir=base_dir,
            allow_code=allow_code, allow_stale=allow_stale, _temp=temp,
        )

        referenced = spipe._referenced_artifact_ids()
        available = [artifact_id for artifact_id in referenced if store.has(artifact_id)]
        for artifact_id in sorted(available):
            store.verify(artifact_id)
        if len(available) < len(referenced):
            logger.info(
                "Thin spipe: %d of %d referenced artifacts unavailable; integrity and "
                "staleness checks defer to pipeline construction.",
                len(referenced) - len(available), len(referenced),
            )
        elif not allow_stale:
            spipe._check_staleness(raise_on_stale=True)

        return spipe

    def save(self, path: str | Path, *, artifacts: str = "fat") -> Path:
        """Write the spipe to `path`.

        A path ending in `.spipe` produces a zip file; any other path produces (or replaces)
        a directory. `artifacts="fat"` (default) embeds every referenced artifact;
        `artifacts="thin"` writes the manifest only, leaving artifact ids resolvable at load
        via `artifact_store=`. A recipe-only spipe with no artifact references writes no
        `artifacts/` directory either way.

        Saving onto the bundle's own backing directory rewrites the manifest in place. A fat
        save there also embeds any referenced artifact the store resolves externally, and a
        thin save there raises `SpipeSaveError` when the directory embeds artifacts, since a
        thin export would have to delete them.

        Args:
            path: Destination file or directory.
            artifacts: `"fat"` or `"thin"`.

        Returns:
            The written path.

        Raises:
            SpipeSaveError: If a directory target exists and is neither empty nor a spipe
                directory, a referenced artifact is unavailable for a fat export, or a thin
                export targets the bundle's own directory while it embeds artifacts.
        """
        if artifacts not in ("fat", "thin"):
            raise SpipeSaveError(f"artifacts must be 'fat' or 'thin'; got {artifacts!r}.")
        path = Path(path)
        referenced = sorted(self._referenced_artifact_ids())
        if artifacts == "fat" and referenced:
            if self._store is None:
                raise SpipeSaveError("This spipe has no artifact store; only artifacts='thin' is possible.")
            for artifact_id in referenced:
                self._store.verify(artifact_id)

        if path.suffix == ".spipe":
            with tempfile.TemporaryDirectory(prefix="spipe-save-") as staging:
                write_manifest(self._manifest, staging)
                if artifacts == "fat" and referenced:
                    self._store.copy_into(Path(staging) / ARTIFACTS_DIR, referenced)
                path.parent.mkdir(parents=True, exist_ok=True)
                pack_zip(staging, path)
            return path

        if self._base_dir is not None and path.exists() \
                and path.resolve() == Path(self._base_dir).resolve():
            # saving onto the backing directory rewrites the manifest in place; a fat save also
            # embeds artifacts the store resolves externally, and a thin save is refused when the
            # directory embeds artifacts since it would have to delete them
            if artifacts == "thin" and (path / ARTIFACTS_DIR).is_dir():
                raise SpipeSaveError(
                    f"A thin export onto the bundle's own directory {path} would delete its "
                    "embedded artifacts; save the thin export to a new path."
                )
            write_manifest(self._manifest, path)
            if artifacts == "fat" and referenced:
                self._store.copy_into(path / ARTIFACTS_DIR, referenced)
            return path

        if path.exists():
            if not path.is_dir():
                raise SpipeSaveError(f"{path} exists and is not a directory.")
            occupied = any(path.iterdir())
            if occupied and not (path / MANIFEST_NAME).exists():
                raise SpipeSaveError(
                    f"{path} exists and is not a spipe directory; refusing to replace it."
                )
            for member in (path / MANIFEST_NAME, path / ARTIFACTS_DIR):
                if member.is_dir():
                    shutil.rmtree(member)
                elif member.exists():
                    member.unlink()
        path.mkdir(parents=True, exist_ok=True)
        write_manifest(self._manifest, path)
        if artifacts == "fat" and referenced:
            self._store.copy_into(path / ARTIFACTS_DIR, referenced)
        return path

    # pipeline construction

    def pipeline(
        self,
        *,
        backend: BackendSpec | str | None = None,
        prefer: str = "frozen",
        verify: str = "strict",
        **pipeline_kwargs,
    ) -> SteeringPipeline:
        """Instantiate a `SteeringPipeline` from this spipe.

        Frozen entries instantiate from their resolution by default; `prefer="recipe"`
        instantiates every entry from its recipe args (forcing re-fits at `steer()`). The
        spipe supplies the model reference and the controls only; backend, device, dtype, and
        `hf_model_kwargs` are the caller's. The caller runs `pipeline.check()` /
        `pipeline.steer()` as normal.

        Args:
            backend: Forwarded to the `SteeringPipeline` constructor.
            prefer: `"frozen"` (default) or `"recipe"`.
            verify: Verification policy for frozen steering artifacts (`"strict"`, `"warn"`,
                or `"off"`), enforced where binding happens at `steer()`.

        Returns:
            The constructed (unsteered) `SteeringPipeline`.

        Raises:
            SpipeFormatError: If a method key resolves to no registered method.
            SpipeCodeRefError: If decoding requires code and the spipe was loaded without
                `allow_code=True`.
            SpipeStaleError: If a deferred staleness check fails (thin bundles).
        """
        if prefer not in ("frozen", "recipe"):
            raise ValueError(f"prefer must be 'frozen' or 'recipe'; got {prefer!r}.")
        if verify not in ("strict", "warn", "off"):
            raise ValueError(f"verify must be 'strict', 'warn', or 'off'; got {verify!r}.")

        if prefer == "frozen" and not self._allow_stale:
            self._check_staleness(raise_on_stale=True)

        ctx = DecodeContext(
            store=self._store,
            allow_code=self._allow_code,
            code_mode="strict",
            verify=verify,
            data_mode="load",
            manifest_records=self._manifest_records,
        )

        controls = []
        for entry in self._manifest["controls"]:
            items = _resolved_items(entry) if prefer == "frozen" else []
            if items:
                for item in items:
                    args = dict(item["args"])
                    if verify != "strict" and item["method"] == "output_control/routed_decoding":
                        args["allow_model_mismatch"] = True
                    if verify != "strict" and item["method"] == "structural_control/load_lora":
                        args["allow_base_mismatch"] = True
                    control = self._instantiate(item["method"], args, ctx)
                    control.enabled = entry["enabled"]
                    controls.append(control)
            else:
                control = self._instantiate(entry["method"], entry["args"], ctx)
                control.enabled = entry["enabled"]
                controls.append(control)

        # controls may hold paths into this spipe's extraction directory; retaining the spipe
        # on each control keeps that directory alive for the pipeline's lifetime
        for control in controls:
            control._spipe_retainer = self

        from steerability.algorithms.core.steering_pipeline import SteeringPipeline

        return SteeringPipeline(
            model_name_or_path=self._manifest["model"]["ref"],
            controls=controls,
            backend=backend,
            **pipeline_kwargs,
        )
