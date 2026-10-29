from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .cluster import CommandError, run_command, run_json_command
from .models import ResolvedRunPlan, ValidationError
from .rhoai_gateway import (
    RHOAI_GATEWAY_NAMESPACE,
    load_rhoai_gateway_configuration,
    rhoai_release_gateway_name,
    wait_for_rhoai_gateway_ready,
)
from .ui import detail, step, success

LLMBATCHGATEWAY_CRD_CANDIDATES = (
    "llmbatchgateways.batch.llm-d.ai",
    "llmbatchgateways.llm-d.ai",
)
LLMBATCHGATEWAY_PREFERRED_VERSION = "v1alpha1"
INFERENCE_OBJECTIVE_CRD_CANDIDATES = (
    "inferenceobjectives.inference.networking.x-k8s.io",
    "inferenceobjectives.llm-d.ai",
)
# llm-d-router EPP discovery order (pkg/epp/server/controller_config.go):
# when both CRDs are installed the EPP prefers llm-d.ai and IGNORES legacy
# resources, so community deployments must create objectives there.
COMMUNITY_INFERENCE_OBJECTIVE_CRD_CANDIDATES = (
    "inferenceobjectives.llm-d.ai",
    "inferenceobjectives.inference.networking.x-k8s.io",
)
INFERENCE_OBJECTIVE_PREFERRED_VERSION = "v1alpha2"
INFERENCE_POOL_CRD_CANDIDATES = (
    "inferencepools.inference.networking.k8s.io",
    "inferencepools.inference.networking.x-k8s.io",
)
GATEWAY_ROUTE_NAMESPACE_LABEL = "llm-d.ai/gateway-route"
BATCH_GATEWAY_LABEL = "benchflow.io/batch-gateway"
BATCH_FILES_PVC_SIZE = "20Gi"
BATCH_FILES_PVC_ACCESS_MODES = ["ReadWriteMany"]

_BATCH_GATEWAY_OPTION_KEYS = {
    "secret_name",
    "db_backend",
    "file_storage",
    "api_server_replicas",
    "processor_replicas",
    "dispatch_mode",
    "implementation",
    "transport",
    "images",
    "interactive_objective",
    "internal_route_auth",
    "request_timeout",
    "max_retries",
    "initial_backoff",
    "max_backoff",
    "inference_objective",
    "async_config",
    "gc_interval",
    "tls",
}
_COMMUNITY_IMAGE_KEYS = ("apiserver", "processor", "gc", "async")
_BATCH_GATEWAY_IMPLEMENTATIONS = ("operator", "community")
_BATCH_GATEWAY_TRANSPORTS = ("redis-sortedset", "sql")
_SYNC_ONLY_OPTION_KEYS = (
    "request_timeout",
    "max_retries",
    "initial_backoff",
    "max_backoff",
)
_DURATION_RE = re.compile(r"^\d+(\.\d+)?(ms|s|m|h)$")
_UPSTREAM_GUIDE_URL = "https://github.com/opendatahub-io/batch-gateway/blob/main/docs/guides/deploy-rhoai.md"


@dataclass(frozen=True, slots=True)
class BatchGatewaySpec:
    """Validated options.batch_gateway configuration for one release."""

    secret_name: str
    db_backend: str
    file_storage: dict[str, Any]
    api_server_replicas: int
    processor_replicas: int
    dispatch_mode: str
    request_timeout: str
    max_retries: int
    initial_backoff: str
    max_backoff: str
    inference_objective: str
    async_config: dict[str, Any]
    gc_interval: str
    tls: dict[str, Any]
    implementation: str = "operator"
    transport: str = "redis-sortedset"
    images: dict[str, str] = field(default_factory=dict)
    interactive_objective: str = ""
    internal_route_auth: bool = True

    @property
    def community_implementation(self) -> bool:
        return self.implementation == "community"

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls.get("enabled"))

    @property
    def uses_fs_storage(self) -> bool:
        return "fs" in self.file_storage

    @property
    def fs_claim_name(self) -> str:
        return str(self.file_storage.get("fs", {}).get("claim_name") or "")

    @property
    def fs_base_path(self) -> str:
        return str(self.file_storage.get("fs", {}).get("base_path") or "")

    @property
    def fs_storage_class_name(self) -> str:
        return str(self.file_storage.get("fs", {}).get("storage_class_name") or "")


def _scoped_name(release_name: str, suffix: str) -> str:
    candidate = f"{release_name}-{suffix}"
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    prefix_length = 63 - len(suffix) - len(digest) - 2
    return f"{release_name[:prefix_length].rstrip('-')}-{digest}-{suffix}"


def _require_nonempty_string(value: Any, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValidationError(f"{field_name} is required")
    return cleaned


def _optional_duration(value: Any, field_name: str, default: str) -> str:
    if value is None:
        return default
    cleaned = str(value).strip()
    if not _DURATION_RE.match(cleaned):
        raise ValidationError(
            f"{field_name} must be a Go duration such as 5m, 30s, or 1h"
        )
    return cleaned


def _positive_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValidationError(f"{field_name} must be a positive integer")
    return parsed


def _parse_file_storage(raw: Any) -> dict[str, Any]:
    if raw is None:
        raise ValidationError("options.batch_gateway.file_storage is required")
    if not isinstance(raw, dict):
        raise ValidationError("options.batch_gateway.file_storage must be a mapping")
    unknown = sorted(set(raw) - {"fs", "s3"})
    if unknown:
        raise ValidationError(
            "options.batch_gateway.file_storage contains unsupported fields: "
            + ", ".join(unknown)
        )
    selected = [key for key in ("fs", "s3") if key in raw]
    if len(selected) != 1:
        raise ValidationError(
            "options.batch_gateway.file_storage must specify exactly one of fs or s3"
        )
    if selected[0] == "s3":
        s3 = raw["s3"]
        if not isinstance(s3, dict) or not s3:
            raise ValidationError(
                "options.batch_gateway.file_storage.s3 must be a non-empty mapping "
                "following the LLMBatchGateway fileStorage.s3 schema"
            )
        return {"s3": deepcopy(s3)}
    fs = raw["fs"]
    if not isinstance(fs, dict):
        raise ValidationError("options.batch_gateway.file_storage.fs must be a mapping")
    unknown = sorted(set(fs) - {"claim_name", "base_path", "storage_class_name"})
    if unknown:
        raise ValidationError(
            "options.batch_gateway.file_storage.fs contains unsupported fields: "
            + ", ".join(unknown)
        )
    return {
        "fs": {
            "claim_name": str(fs.get("claim_name") or "").strip(),
            "base_path": str(fs.get("base_path") or "/tmp/batch-gateway").strip(),
            "storage_class_name": str(fs.get("storage_class_name") or "").strip(),
        }
    }


def _parse_async_config(
    raw: Any, dispatch_mode: str, *, derived: bool = False
) -> dict[str, Any]:
    if dispatch_mode == "async":
        if derived:
            if raw:
                raise ValidationError(
                    "options.batch_gateway.async_config is derived by BenchFlow "
                    "for implementation: community; remove it from the profile"
                )
            return {}
        if not isinstance(raw, dict) or not raw:
            raise ValidationError(
                "options.batch_gateway.async_config is required for dispatch_mode: async"
            )
        return deepcopy(raw)
    if raw:
        raise ValidationError(
            "options.batch_gateway.async_config applies only to dispatch_mode: async"
        )
    return {}


def _parse_implementation(raw: Any) -> str:
    implementation = str(raw or "operator").strip()
    if implementation not in _BATCH_GATEWAY_IMPLEMENTATIONS:
        raise ValidationError(
            "options.batch_gateway.implementation must be one of: "
            + ", ".join(_BATCH_GATEWAY_IMPLEMENTATIONS)
        )
    return implementation


def _parse_transport(raw: Any) -> str:
    transport = str(raw or "redis-sortedset").strip()
    if transport not in _BATCH_GATEWAY_TRANSPORTS:
        raise ValidationError(
            "options.batch_gateway.transport must be one of: "
            + ", ".join(_BATCH_GATEWAY_TRANSPORTS)
        )
    return transport


def _parse_images(raw: Any, implementation: str) -> dict[str, str]:
    if implementation != "community":
        if raw:
            raise ValidationError(
                "options.batch_gateway.images applies only to "
                "implementation: community"
            )
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(
            "options.batch_gateway.images is required for implementation: "
            "community and must be a mapping of component image references"
        )
    unknown = sorted(set(raw) - set(_COMMUNITY_IMAGE_KEYS))
    if unknown:
        raise ValidationError(
            "options.batch_gateway.images contains unsupported fields: "
            + ", ".join(unknown)
        )
    images: dict[str, str] = {}
    for key in _COMMUNITY_IMAGE_KEYS:
        value = str(raw.get(key) or "").strip()
        if not value:
            raise ValidationError(
                f"options.batch_gateway.images.{key} is required for "
                "implementation: community; pin a digest for reproducibility"
            )
        images[key] = value
    return images


def _parse_tls(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError("options.batch_gateway.tls must be a mapping")
    if not isinstance(raw.get("enabled"), bool):
        raise ValidationError("options.batch_gateway.tls.enabled must be a boolean")
    return deepcopy(raw)


def parse_batch_gateway_options(raw: Any) -> dict[str, Any]:
    """Validate and normalize the profile options.batch_gateway mapping.

    Field names follow the LLMBatchGateway CR documented in the upstream
    deploy-rhoai.md guide; nested blocks the guide defines verbatim
    (file_storage.s3, async_config, tls) pass through unchanged.
    """
    if not isinstance(raw, dict):
        raise ValidationError("options.batch_gateway must be a mapping")
    unknown = sorted(set(raw) - _BATCH_GATEWAY_OPTION_KEYS)
    if unknown:
        raise ValidationError(
            "options.batch_gateway contains unsupported fields: " + ", ".join(unknown)
        )

    secret_name = _require_nonempty_string(
        raw.get("secret_name"), "options.batch_gateway.secret_name"
    )
    db_backend = str(raw.get("db_backend") or "postgresql").strip()
    if db_backend not in {"postgresql", "redis"}:
        raise ValidationError(
            "options.batch_gateway.db_backend must be 'postgresql' or 'redis'"
        )
    dispatch_mode = str(raw.get("dispatch_mode") or "sync").strip()
    if dispatch_mode not in {"sync", "async"}:
        raise ValidationError(
            "options.batch_gateway.dispatch_mode must be 'sync' or 'async'"
        )
    implementation = _parse_implementation(raw.get("implementation"))
    transport = _parse_transport(raw.get("transport"))
    if transport == "sql":
        if implementation != "community":
            raise ValidationError(
                "options.batch_gateway.transport: sql requires implementation: "
                "community — the RHOAI 3.5 LLMBatchGateway CRD cannot express the "
                "SQL queue transport"
            )
        if dispatch_mode != "async":
            raise ValidationError(
                "options.batch_gateway.transport: sql requires dispatch_mode: async"
            )
        if db_backend != "postgresql":
            raise ValidationError(
                "options.batch_gateway.transport: sql requires db_backend: "
                "postgresql — the SQL queues live in the batch gateway database"
            )
    if implementation == "community" and dispatch_mode != "async":
        raise ValidationError(
            "options.batch_gateway.implementation: community supports "
            "dispatch_mode: async only; sync dispatch is covered by the "
            "operator path"
        )
    images = _parse_images(raw.get("images"), implementation)
    interactive_objective = str(raw.get("interactive_objective") or "").strip()
    internal_route_auth = raw.get("internal_route_auth")
    if internal_route_auth is None:
        internal_route_auth = True
    if not isinstance(internal_route_auth, bool):
        raise ValidationError(
            "options.batch_gateway.internal_route_auth must be a boolean"
        )

    sync_only_set = [key for key in _SYNC_ONLY_OPTION_KEYS if raw.get(key) is not None]
    if dispatch_mode == "async" and sync_only_set:
        raise ValidationError(
            "options.batch_gateway "
            + ", ".join(sync_only_set)
            + " apply only to dispatch_mode: sync"
        )

    file_storage = _parse_file_storage(raw.get("file_storage"))
    async_config = _parse_async_config(
        raw.get("async_config"), dispatch_mode, derived=implementation == "community"
    )
    tls = _parse_tls(raw.get("tls"))

    gc_interval = raw.get("gc_interval")
    if gc_interval is None:
        gc_interval = "30m"
    else:
        gc_interval = str(gc_interval).strip()
        if gc_interval and not _DURATION_RE.match(gc_interval):
            raise ValidationError(
                "options.batch_gateway.gc_interval must be a Go duration such as 30m"
            )

    return {
        "secret_name": secret_name,
        "db_backend": db_backend,
        "file_storage": file_storage,
        "api_server_replicas": _positive_int(
            raw.get("api_server_replicas"),
            "options.batch_gateway.api_server_replicas",
            1,
        ),
        "processor_replicas": _positive_int(
            raw.get("processor_replicas"), "options.batch_gateway.processor_replicas", 1
        ),
        "dispatch_mode": dispatch_mode,
        "request_timeout": _optional_duration(
            raw.get("request_timeout"), "options.batch_gateway.request_timeout", "5m"
        ),
        "max_retries": _positive_int(
            raw.get("max_retries"),
            "options.batch_gateway.max_retries",
            3,
        ),
        "initial_backoff": _optional_duration(
            raw.get("initial_backoff"), "options.batch_gateway.initial_backoff", "1s"
        ),
        "max_backoff": _optional_duration(
            raw.get("max_backoff"), "options.batch_gateway.max_backoff", "60s"
        ),
        "inference_objective": str(raw.get("inference_objective") or "").strip(),
        "async_config": async_config,
        "gc_interval": gc_interval,
        "tls": tls,
        "implementation": implementation,
        "transport": transport,
        "images": images,
        "interactive_objective": interactive_objective,
        "internal_route_auth": internal_route_auth,
    }


def rhoai_batch_gateway_requested(plan: ResolvedRunPlan) -> bool:
    return plan.deployment.options.get("batch_gateway") is not None


def rhoai_batch_gateway_spec(plan: ResolvedRunPlan) -> BatchGatewaySpec | None:
    raw = plan.deployment.options.get("batch_gateway")
    if raw is None:
        return None
    if plan.deployment.platform != "rhoai" or plan.deployment.mode == "isvc":
        raise ValidationError(
            "options.batch_gateway is supported only for rhoai LLMInferenceService "
            "deployments"
        )
    parsed = parse_batch_gateway_options(raw)
    release_name = plan.deployment.release_name
    inference_objective = parsed["inference_objective"] or _scoped_name(
        release_name, "batch-sheddable"
    )
    file_storage = parsed["file_storage"]
    if "fs" in file_storage and not file_storage["fs"]["claim_name"]:
        file_storage = {
            "fs": {
                **file_storage["fs"],
                "claim_name": _scoped_name(release_name, "batch-gateway-files"),
            }
        }
    return BatchGatewaySpec(
        secret_name=parsed["secret_name"],
        db_backend=parsed["db_backend"],
        file_storage=file_storage,
        api_server_replicas=parsed["api_server_replicas"],
        processor_replicas=parsed["processor_replicas"],
        dispatch_mode=parsed["dispatch_mode"],
        request_timeout=parsed["request_timeout"],
        max_retries=parsed["max_retries"],
        initial_backoff=parsed["initial_backoff"],
        max_backoff=parsed["max_backoff"],
        inference_objective=inference_objective,
        async_config=parsed["async_config"],
        gc_interval=parsed["gc_interval"],
        tls=parsed["tls"],
        implementation=parsed["implementation"],
        transport=parsed["transport"],
        images=parsed["images"],
        interactive_objective=parsed["interactive_objective"],
        internal_route_auth=parsed["internal_route_auth"],
    )


def llmbatchgateway_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "batch-gateway")


def rhoai_batch_internal_gateway_name(plan: ResolvedRunPlan) -> str:
    """ClusterIP-only internal Gateway name, release-scoped.

    Istio derives the gateway Deployment label from ``<gateway>-<class>``;
    this stays within the 63-character label limit for the RHOAI
    GatewayClass used by BenchFlow.
    """
    return f"{rhoai_release_gateway_name(plan)}-batch"


def batch_llm_route_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "batch-llm-route")


def batch_route_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "batch-route")


def batch_llm_route_authpolicy_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "batch-llm-route-auth")


def batch_route_authpolicy_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "batch-route-auth")


def batch_destinationrule_name(plan: ResolvedRunPlan) -> str:
    return f"{rhoai_release_gateway_name(plan)}-batch-tls"


def _batch_labels(plan: ResolvedRunPlan, component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "benchflow",
        "app.kubernetes.io/managed-by": "benchflow",
        "app.kubernetes.io/component": component,
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/release": plan.deployment.release_name,
        BATCH_GATEWAY_LABEL: "true",
    }


def _batch_release_selector(plan: ResolvedRunPlan) -> str:
    return (
        f"{BATCH_GATEWAY_LABEL}=true,benchflow.io/release="
        f"{plan.deployment.release_name}"
    )


# ---------------------------------------------------------------------------
# Cluster discovery


def _crd_json(kubectl_cmd: str, crd_name: str) -> dict[str, Any] | None:
    result = run_command(
        [kubectl_cmd, "get", "crd", crd_name, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _crd_api_version(
    kubectl_cmd: str,
    candidates: tuple[str, ...],
    preferred_version: str,
    *,
    label: str,
) -> str:
    """Resolve the served apiVersion for a CRD whose group drifted upstream."""
    for crd_name in candidates:
        payload = _crd_json(kubectl_cmd, crd_name)
        if payload is None:
            continue
        versions = [
            str(version.get("name") or "")
            for version in (payload.get("spec") or {}).get("versions") or []
            if isinstance(version, dict) and version.get("served")
        ]
        if not versions:
            continue
        group = crd_name.split(".", 1)[1]
        version = preferred_version if preferred_version in versions else versions[0]
        return f"{group}/{version}"
    raise CommandError(
        f"{label} CRD was not found on the cluster; expected one of: "
        + ", ".join(candidates)
    )


def llmbatchgateway_api_version(kubectl_cmd: str) -> str:
    return _crd_api_version(
        kubectl_cmd,
        LLMBATCHGATEWAY_CRD_CANDIDATES,
        LLMBATCHGATEWAY_PREFERRED_VERSION,
        label="LLMBatchGateway",
    )


def inference_objective_api_version(kubectl_cmd: str) -> str:
    return _crd_api_version(
        kubectl_cmd,
        INFERENCE_OBJECTIVE_CRD_CANDIDATES,
        INFERENCE_OBJECTIVE_PREFERRED_VERSION,
        label="InferenceObjective",
    )


def community_inference_objective_api_version(kubectl_cmd: str) -> str:
    """Mirror the llm-d-router EPP CRD discovery order: prefer llm-d.ai.

    The community llm-d-router EPP resolves request objectives from the
    llm-d.ai InferenceObjective CRD when both it and the legacy
    inference.networking.x-k8s.io CRD are installed (it logs "EPP will
    prefer the new group and IGNORE legacy resources"). BenchFlow must
    create the objectives in the group the running EPP actually watches.
    """
    return _crd_api_version(
        kubectl_cmd,
        COMMUNITY_INFERENCE_OBJECTIVE_CRD_CANDIDATES,
        INFERENCE_OBJECTIVE_PREFERRED_VERSION,
        label="InferenceObjective",
    )


def inference_pool_group(kubectl_cmd: str) -> str:
    for crd_name in INFERENCE_POOL_CRD_CANDIDATES:
        if _crd_json(kubectl_cmd, crd_name) is not None:
            return crd_name.split(".", 1)[1]
    raise CommandError(
        "InferencePool CRD was not found on the cluster; expected one of: "
        + ", ".join(INFERENCE_POOL_CRD_CANDIDATES)
    )


def _owner_references(item: dict[str, Any]) -> list[dict[str, Any]]:
    owners = ((item.get("metadata") or {}).get("ownerReferences")) or []
    return [owner for owner in owners if isinstance(owner, dict)]


def discover_inference_pool_name(
    kubectl_cmd: str,
    *,
    namespace: str,
    isvc_name: str,
    timeout_seconds: int,
) -> str:
    """Find the InferencePool owned by one LLMInferenceService release."""
    deadline = time.time() + timeout_seconds
    conventional_name = f"{isvc_name}-inference-pool"
    while time.time() < deadline:
        payload = run_json_command(
            [kubectl_cmd, "get", "inferencepool", "-n", namespace, "-o", "json"]
        )
        items = payload.get("items") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str((item.get("metadata") or {}).get("name") or "")
            if any(owner.get("name") == isvc_name for owner in _owner_references(item)):
                return name
        for item in items:
            if not isinstance(item, dict):
                continue
            if str((item.get("metadata") or {}).get("name") or "") == conventional_name:
                return conventional_name
        time.sleep(10)
    raise CommandError(
        f"timed out waiting for the InferencePool owned by {namespace}/{isvc_name}"
    )


def discover_batch_internal_gateway_service(
    kubectl_cmd: str,
    *,
    gateway_name: str,
    timeout_seconds: int,
) -> str:
    label_selector = f"gateway.networking.k8s.io/gateway-name={gateway_name}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "svc",
                "-n",
                RHOAI_GATEWAY_NAMESPACE,
                "-l",
                label_selector,
                "-o",
                "json",
            ]
        )
        items = payload.get("items") or []
        for item in items:
            if isinstance(item, dict):
                name = str((item.get("metadata") or {}).get("name") or "")
                if name:
                    return name
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for the Service of Gateway {gateway_name} "
        f"in {RHOAI_GATEWAY_NAMESPACE}"
    )


def discover_batch_apiserver_service(
    kubectl_cmd: str,
    *,
    namespace: str,
    cr_name: str,
    timeout_seconds: int,
) -> str:
    deadline = time.time() + timeout_seconds
    conventional_names = (f"{cr_name}-apiserver", "batch-gateway-apiserver")
    while time.time() < deadline:
        payload = run_json_command(
            [kubectl_cmd, "get", "svc", "-n", namespace, "-o", "json"]
        )
        items = payload.get("items") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str((item.get("metadata") or {}).get("name") or "")
            if not name:
                continue
            owned = any(
                owner.get("name") == cr_name for owner in _owner_references(item)
            )
            if owned and "apiserver" in name:
                return name
        for candidate in conventional_names:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str((item.get("metadata") or {}).get("name") or "") == candidate:
                    return candidate
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for the batch gateway apiserver Service in {namespace}"
    )


def wait_for_llmbatchgateway_ready(
    kubectl_cmd: str,
    *,
    namespace: str,
    name: str,
    timeout_seconds: int,
) -> None:
    deadline = time.time() + timeout_seconds
    last_reason = ""
    while time.time() < deadline:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "llmbatchgateway",
                name,
                "-n",
                namespace,
                "-o",
                "json",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                payload = {}
            conditions = (payload.get("status") or {}).get("conditions") or []
            for condition in conditions:
                if not isinstance(condition, dict):
                    continue
                if condition.get("type") != "Ready":
                    continue
                if condition.get("status") == "True":
                    return
                reason = str(
                    condition.get("reason") or condition.get("message") or "waiting"
                )
                if reason != last_reason:
                    detail(f"LLMBatchGateway {name} not ready yet: {reason}")
                    last_reason = reason
                break
        time.sleep(10)
    raise CommandError(
        f"timed out waiting for LLMBatchGateway {namespace}/{name} to become Ready"
    )


def rhoai_batch_endpoint_url(kubectl_cmd: str, plan: ResolvedRunPlan) -> str:
    """External batch API base URL served by the release Gateway."""
    gateway_name = rhoai_release_gateway_name(plan)
    payload = run_json_command(
        [
            kubectl_cmd,
            "get",
            "gateway",
            gateway_name,
            "-n",
            RHOAI_GATEWAY_NAMESPACE,
            "-o",
            "json",
        ]
    )
    # The OpenShift router routes by SNI, so the listener hostname is the
    # client-usable address; the status address is the raw load balancer
    # behind it and does not route on its own.
    for listener in (payload.get("spec") or {}).get("listeners") or []:
        if not isinstance(listener, dict):
            continue
        hostname = str(listener.get("hostname") or "").strip()
        if hostname:
            return f"https://{hostname}".rstrip("/")
    addresses = (payload.get("status") or {}).get("addresses") or []
    for address in addresses:
        if not isinstance(address, dict):
            continue
        value = str(address.get("value") or "").strip()
        if value:
            if value.startswith(("http://", "https://")):
                return value.rstrip("/")
            return f"https://{value}".rstrip("/")
    raise CommandError(
        f"Gateway {gateway_name} in {RHOAI_GATEWAY_NAMESPACE} does not have a "
        "status address yet"
    )


# ---------------------------------------------------------------------------
# Manifest rendering


def render_batch_internal_gateway_manifest(
    plan: ResolvedRunPlan, gateway_config: Any
) -> dict[str, Any]:
    """ClusterIP-only Gateway the batch processor dispatches through."""
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {
            "name": rhoai_batch_internal_gateway_name(plan),
            "namespace": RHOAI_GATEWAY_NAMESPACE,
            "annotations": {
                "networking.istio.io/service-type": "ClusterIP",
            },
            "labels": {
                **_batch_labels(plan, "batch-internal-gateway"),
                **gateway_config.labels,
            },
        },
        "spec": {
            "gatewayClassName": gateway_config.gateway_class_name,
            "listeners": [
                {
                    "name": "http",
                    "port": 80,
                    "protocol": "HTTP",
                    "allowedRoutes": {
                        "namespaces": {
                            "from": "Selector",
                            "selector": {
                                "matchLabels": {GATEWAY_ROUTE_NAMESPACE_LABEL: "true"}
                            },
                        }
                    },
                }
            ],
        },
    }


def render_batch_inference_objective_manifest(
    plan: ResolvedRunPlan,
    *,
    spec: BatchGatewaySpec,
    api_version: str,
    pool_group: str,
    pool_name: str,
    name: str | None = None,
    priority: int = -1,
    component: str = "batch-inference-objective",
) -> dict[str, Any]:
    """Objective CR the EPP resolves to a priority band.

    The batch objective defaults to priority -1 (sheddable). Callers that
    render an interactive objective pass name/priority/component explicitly.
    """
    return {
        "apiVersion": api_version,
        "kind": "InferenceObjective",
        "metadata": {
            "name": name or spec.inference_objective,
            "namespace": plan.deployment.namespace,
            "labels": _batch_labels(plan, component),
        },
        "spec": {
            "priority": priority,
            "poolRef": {"group": pool_group, "name": pool_name},
        },
    }


def render_batch_llm_route_manifest(
    plan: ResolvedRunPlan,
    *,
    pool_group: str,
    pool_name: str,
) -> dict[str, Any]:
    """Route /{namespace}/{isvc}/v1/* on the internal Gateway to the pool."""
    namespace = plan.deployment.namespace
    isvc_name = plan.deployment.release_name
    rules = []
    for endpoint in ("v1/completions", "v1/chat/completions"):
        rules.append(
            {
                "matches": [
                    {
                        "path": {
                            "type": "PathPrefix",
                            "value": f"/{namespace}/{isvc_name}/{endpoint}",
                        }
                    }
                ],
                "filters": [
                    {
                        "type": "URLRewrite",
                        "urlRewrite": {
                            "path": {
                                "type": "ReplacePrefixMatch",
                                "replacePrefixMatch": f"/{endpoint}",
                            }
                        },
                    }
                ],
                "backendRefs": [
                    {
                        "group": pool_group,
                        "kind": "InferencePool",
                        "name": pool_name,
                    }
                ],
            }
        )
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": batch_llm_route_name(plan),
            "namespace": namespace,
            "labels": _batch_labels(plan, "batch-llm-route"),
        },
        "spec": {
            "parentRefs": [
                {
                    "name": rhoai_batch_internal_gateway_name(plan),
                    "namespace": RHOAI_GATEWAY_NAMESPACE,
                }
            ],
            "rules": rules,
        },
    }


def render_batch_llm_route_authpolicy(plan: ResolvedRunPlan) -> dict[str, Any]:
    """Token authentication plus model-access SubjectAccessReview."""
    return {
        "apiVersion": "kuadrant.io/v1",
        "kind": "AuthPolicy",
        "metadata": {
            "name": batch_llm_route_authpolicy_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": _batch_labels(plan, "batch-llm-route-auth"),
        },
        "spec": {
            "targetRef": {
                "group": "gateway.networking.k8s.io",
                "kind": "HTTPRoute",
                "name": batch_llm_route_name(plan),
            },
            "rules": {
                "authentication": {
                    "kubernetes-user": {
                        "kubernetesTokenReview": {
                            "audiences": ["https://kubernetes.default.svc"]
                        }
                    }
                },
                "authorization": {
                    "model-access": {
                        "kubernetesSubjectAccessReview": {
                            "user": {"expression": "auth.identity.user.username"},
                            "authorizationGroups": {
                                "expression": "auth.identity.user.groups"
                            },
                            "resourceAttributes": {
                                "group": {"value": "serving.kserve.io"},
                                "resource": {"value": "llminferenceservices"},
                                "namespace": {
                                    "expression": 'request.path.split("/")[1]'
                                },
                                "name": {"expression": 'request.path.split("/")[2]'},
                                "verb": {"value": "get"},
                            },
                        }
                    }
                },
            },
        },
    }


def _inject_release_scoped_async_defaults(
    async_config: dict[str, Any],
    *,
    model_url: str,
    pool_name: str,
    namespace: str,
) -> dict[str, Any]:
    """Fill release-derived identity fields the profile cannot know.

    Queues get a default name, the internal-gateway base URL, and gate
    parameters when the profile does not pin them explicitly.
    """
    config = deepcopy(async_config)
    redis_config = config.get("redis")
    if not isinstance(redis_config, dict):
        return config
    queues = redis_config.get("queuesConfig")
    if not isinstance(queues, list):
        return config
    for queue in queues:
        if not isinstance(queue, dict):
            continue
        queue.setdefault("name", f"llm-d-async:requests:{pool_name}")
        queue.setdefault("igwBaseURL", model_url)
        gate_params = queue.get("gateParams")
        if not isinstance(gate_params, dict):
            continue
        if str(queue.get("gateType") or "").strip() != "prometheus-budget":
            continue
        gate_params.setdefault("pool", pool_name)
        gate_params.setdefault("namespace", namespace)
    return config


def render_llmbatchgateway_manifest(
    plan: ResolvedRunPlan,
    *,
    spec: BatchGatewaySpec,
    api_version: str,
    pool_name: str,
    model_url: str,
) -> dict[str, Any]:
    model_gateway: dict[str, Any] = {
        "inferenceObjective": spec.inference_objective,
    }
    if spec.dispatch_mode == "async":
        model_gateway["inferencePoolName"] = pool_name
    else:
        model_gateway.update(
            {
                "url": model_url,
                "requestTimeout": spec.request_timeout,
                "maxRetries": spec.max_retries,
                "initialBackoff": spec.initial_backoff,
                "maxBackoff": spec.max_backoff,
            }
        )

    processor: dict[str, Any] = {
        "replicas": spec.processor_replicas,
        "modelGateways": {plan.model.name: model_gateway},
    }
    if spec.dispatch_mode == "async":
        processor["dispatchMode"] = "async"
        processor["asyncConfig"] = _inject_release_scoped_async_defaults(
            spec.async_config,
            model_url=model_url,
            pool_name=pool_name,
            namespace=plan.deployment.namespace,
        )

    if spec.uses_fs_storage:
        file_storage = {
            "fs": {
                "basePath": spec.fs_base_path,
                "claimName": spec.fs_claim_name,
            }
        }
    else:
        file_storage = {"s3": deepcopy(spec.file_storage.get("s3") or {})}

    manifest: dict[str, Any] = {
        "apiVersion": api_version,
        "kind": "LLMBatchGateway",
        "metadata": {
            "name": llmbatchgateway_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": _batch_labels(plan, "batch-gateway"),
        },
        "spec": {
            "secretRef": {"name": spec.secret_name},
            "dbBackend": spec.db_backend,
            "fileStorage": file_storage,
            "apiServer": {"replicas": spec.api_server_replicas},
            "processor": processor,
        },
    }
    if spec.gc_interval:
        manifest["spec"]["gc"] = {"interval": spec.gc_interval}
    if spec.tls:
        manifest["spec"]["tls"] = deepcopy(spec.tls)
    return manifest


# ---------------------------------------------------------------------------
# Community batch gateway rendering (llm-d/llm-d-batch-gateway images)
#
# The RHOAI 3.5 LLMBatchGateway CRD cannot express the SQL queue transport or
# in-flight eviction, so BenchFlow renders the community components directly,
# mirroring the upstream charts/batch-gateway templates (main branch) as
# Python dicts. Component names reuse the operator convention
# (<release>-batch-gateway-<component>) so service discovery and cleanup work
# identically on both paths.


def community_component_name(plan: ResolvedRunPlan, component: str) -> str:
    return _scoped_name(plan.deployment.release_name, f"batch-gateway-{component}")


def llm_d_async_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "llm-d-async")


def _community_labels(plan: ResolvedRunPlan, component: str) -> dict[str, str]:
    return {
        **_batch_labels(plan, component),
        "app.kubernetes.io/instance": plan.deployment.release_name,
    }


def _community_selector(plan: ResolvedRunPlan, component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "benchflow",
        "app.kubernetes.io/component": component,
        "benchflow.io/release": plan.deployment.release_name,
    }


def _community_selector_flat(plan: ResolvedRunPlan, component: str) -> str:
    return ",".join(
        f"{key}={value}"
        for key, value in sorted(_community_selector(plan, component).items())
    )


def _community_pod_security_context() -> dict[str, Any]:
    return {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def _community_container_security_context() -> dict[str, Any]:
    return {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }


_COMMUNITY_FILES_BASE_PATH = "/tmp/batch-gateway"
_COMMUNITY_PROCESSOR_WORK_DIR = "/var/lib/batch-gateway/processor"
_COMMUNITY_PROCESSOR_WORK_DIR_SIZE = "10Gi"


def _community_file_client(spec: BatchGatewaySpec) -> dict[str, Any]:
    s3 = deepcopy(spec.file_storage.get("s3") or {})
    return {
        "type": "fs" if spec.uses_fs_storage else "s3",
        "fs": {"base_path": spec.fs_base_path or _COMMUNITY_FILES_BASE_PATH},
        "s3": {
            "region": str(s3.get("region") or ""),
            "bucket": str(s3.get("bucket") or ""),
            "endpoint": str(s3.get("endpoint") or ""),
            "access_key_id": str(s3.get("access_key_id") or ""),
            "prefix": str(s3.get("prefix") or ""),
            "use_path_style": bool(s3.get("use_path_style") or False),
            "auto_create_bucket": bool(s3.get("auto_create_bucket", True)),
        },
        "retry": {
            "max_retries": 3,
            "initial_backoff": "1s",
            "max_backoff": "10s",
        },
    }


def _community_apiserver_config(spec: BatchGatewaySpec) -> dict[str, Any]:
    return {
        "host": "",
        "port": "8000",
        "observability_port": "8081",
        "input_headers": {"tenant": "X-MaaS-Username"},
        "read_header_timeout_seconds": 10,
        "read_timeout_seconds": 900,
        "write_timeout_seconds": 120,
        "idle_timeout_seconds": 90,
        "api_shutdown_timeout_seconds": 60,
        "observability_shutdown_timeout_seconds": 60,
        "db_client": {"type": spec.db_backend},
        "file_client": _community_file_client(spec),
        "batch_api": {
            "batch_event_ttl_seconds": 2592000,
            "pass_through_headers": ["Authorization"],
        },
        "file_api": {
            "default_expiration_seconds": 7776000,
            "max_size_bytes": 209715200,
            "max_line_count": 50000,
        },
        "enable_pprof": False,
        "otel": {"postgresql_tracing": False},
    }


def _community_processor_config(
    plan: ResolvedRunPlan,
    spec: BatchGatewaySpec,
    *,
    pool_name: str,
) -> dict[str, Any]:
    request_queue = f"llm-d-async:requests:{pool_name}"
    result_queue = f"llm-d-async:results:{pool_name}"
    async_dispatch: dict[str, Any] = {
        "transport": spec.transport,
        "result_poll_timeout": "30s",
        "poll_interval": "1s",
        "models": {
            plan.model.name: {
                "inference_pool_name": pool_name,
                "inference_objective": spec.inference_objective,
                "request_queue_name": request_queue,
                "result_queue_name": result_queue,
            }
        },
    }
    if spec.transport == "sql":
        async_dispatch["max_connections"] = 8
    return {
        "addr": ":9090",
        "task_wait_time": "1s",
        "poll_interval": "5s",
        "num_workers": 5,
        "work_dir_size_limit": _COMMUNITY_PROCESSOR_WORK_DIR_SIZE,
        "input_file_disk_budget_percent": 10,
        "input_file_max_size_bytes": 209715200,
        "concurrency": {
            "global": 60,
            "per_endpoint": 30,
            "recovery": 5,
            "aimd": {
                "enabled": True,
                "min": 5,
                "backoff_factor": 0.5,
                "additive_increase": 1,
            },
        },
        "terminate_on_observability_failure": False,
        "shutdown_timeout": "30s",
        "work_dir": _COMMUNITY_PROCESSOR_WORK_DIR,
        "db_client": {"type": spec.db_backend},
        "progress_update_interval": "15s",
        "dispatch_mode": "async",
        "async_dispatch": async_dispatch,
        "resumable": {
            "lease_duration": "5m",
            "finalize_concurrency": 16,
            "finalize_timeout": "30m",
        },
        "default_output_expiration_seconds": 7776000,
        "file_client": _community_file_client(spec),
        "queue_time_bucket": {"start": 0.1, "factor": 2, "count": 10},
        "process_time_bucket": {"start": 0.1, "factor": 2, "count": 15},
        "e2e_latency_bucket": {"start": 1, "factor": 3, "count": 12},
        "enable_pprof": False,
        "otel": {"postgresql_tracing": False},
    }


def _community_gc_config(
    plan: ResolvedRunPlan,
    spec: BatchGatewaySpec,
) -> dict[str, Any]:
    return {
        "dry_run": False,
        "collector": {
            "interval": spec.gc_interval or "30m",
            "max_concurrency": 10,
        },
        "metrics_addr": ":9091",
        "db_client": {"type": spec.db_backend},
        "file_client": _community_file_client(spec),
        "reconciler": {
            "enabled": True,
            "interval": "60m",
            "processor_label_selector": _community_selector_flat(
                plan, "batch-gateway-processor"
            ),
            "processor_statefulset": community_component_name(
                plan, "processor"
            ),
        },
    }


def _community_secret_volumes(spec: BatchGatewaySpec) -> list[dict[str, Any]]:
    return [
        {
            "name": "secrets",
            "projected": {"sources": [{"secret": {"name": spec.secret_name}}]},
        }
    ]


def _community_secret_mounts() -> list[dict[str, Any]]:
    return [{"name": "secrets", "mountPath": "/etc/.secrets", "readOnly": True}]


def _community_files_volumes(spec: BatchGatewaySpec) -> list[dict[str, Any]]:
    if not spec.uses_fs_storage:
        return []
    return [
        {
            "name": "files-storage",
            "persistentVolumeClaim": {"claimName": spec.fs_claim_name},
        }
    ]


def _community_files_mounts(spec: BatchGatewaySpec) -> list[dict[str, Any]]:
    if not spec.uses_fs_storage:
        return []
    return [
        {
            "name": "files-storage",
            "mountPath": spec.fs_base_path or _COMMUNITY_FILES_BASE_PATH,
        }
    ]


def _community_config_configmap(
    name: str,
    labels: dict[str, str],
    namespace: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "data": {
            "config.yaml": yaml.safe_dump(config, sort_keys=False),
        },
    }


def _community_service_account(
    name: str, labels: dict[str, str], namespace: str
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "automountServiceAccountToken": True,
    }


def render_community_apiserver_manifest(
    plan: ResolvedRunPlan, spec: BatchGatewaySpec
) -> dict[str, Any]:
    name = community_component_name(plan, "apiserver")
    labels = _community_labels(plan, "batch-gateway-apiserver")
    config = _community_apiserver_config(spec)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": plan.deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": spec.api_server_replicas,
            "selector": {"matchLabels": _community_selector(
                plan, "batch-gateway-apiserver"
            )},
            "template": {
                "metadata": {
                    "annotations": {
                        "checksum/config": hashlib.sha256(
                            yaml.safe_dump(config, sort_keys=False).encode("utf-8")
                        ).hexdigest(),
                    },
                    "labels": _community_labels(plan, "batch-gateway-apiserver"),
                },
                "spec": {
                    "serviceAccountName": name,
                    "securityContext": _community_pod_security_context(),
                    "terminationGracePeriodSeconds": 120,
                    "containers": [
                        {
                            "name": "apiserver",
                            "image": spec.images["apiserver"],
                            "imagePullPolicy": "Always",
                            "securityContext": _community_container_security_context(),
                            "args": [
                                "--config=/etc/config/config.yaml",
                                "--v=2",
                            ],
                            "ports": [
                                {"name": "http", "containerPort": 8000},
                                {"name": "observability", "containerPort": 8081},
                            ],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": "observability",
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": "observability",
                                },
                                "initialDelaySeconds": 15,
                                "periodSeconds": 20,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                            "resources": {
                                "limits": {"cpu": "1", "memory": "512Mi"},
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "config",
                                    "mountPath": "/etc/config",
                                    "readOnly": True,
                                },
                                *_community_files_mounts(spec),
                                *_community_secret_mounts(),
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "config", "configMap": {"name": f"{name}-config"}},
                        *_community_files_volumes(spec),
                        *_community_secret_volumes(spec),
                    ],
                },
            },
        },
    }


def render_community_processor_manifest(
    plan: ResolvedRunPlan,
    spec: BatchGatewaySpec,
    *,
    pool_name: str,
) -> dict[str, Any]:
    name = community_component_name(plan, "processor")
    labels = _community_labels(plan, "batch-gateway-processor")
    config = _community_processor_config(plan, spec, pool_name=pool_name)
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {
            "name": name,
            "namespace": plan.deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": spec.processor_replicas,
            "serviceName": name,
            "podManagementPolicy": "Parallel",
            "selector": {"matchLabels": _community_selector(
                plan, "batch-gateway-processor"
            )},
            "template": {
                "metadata": {
                    "annotations": {
                        "checksum/config": hashlib.sha256(
                            yaml.safe_dump(config, sort_keys=False).encode("utf-8")
                        ).hexdigest(),
                    },
                    "labels": _community_labels(plan, "batch-gateway-processor"),
                },
                "spec": {
                    "serviceAccountName": name,
                    "securityContext": _community_pod_security_context(),
                    "terminationGracePeriodSeconds": 60,
                    "containers": [
                        {
                            "name": "processor",
                            "image": spec.images["processor"],
                            "imagePullPolicy": "Always",
                            "securityContext": _community_container_security_context(),
                            "args": [
                                "--config=/etc/config/config.yaml",
                                "--v=2",
                            ],
                            "ports": [{"name": "metrics", "containerPort": 9090}],
                            "readinessProbe": {
                                "httpGet": {"path": "/ready", "port": "metrics"},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": "metrics"},
                                "initialDelaySeconds": 15,
                                "periodSeconds": 20,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                            "resources": {
                                "limits": {"cpu": "2", "memory": "1Gi"},
                                "requests": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "config",
                                    "mountPath": "/etc/config",
                                    "readOnly": True,
                                },
                                {
                                    "name": "workdir",
                                    "mountPath": _COMMUNITY_PROCESSOR_WORK_DIR,
                                },
                                *_community_files_mounts(spec),
                                *_community_secret_mounts(),
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "config", "configMap": {"name": f"{name}-config"}},
                        {
                            "name": "workdir",
                            "emptyDir": {"sizeLimit": _COMMUNITY_PROCESSOR_WORK_DIR_SIZE},
                        },
                        *_community_files_volumes(spec),
                        *_community_secret_volumes(spec),
                    ],
                },
            },
        },
    }


def render_community_gc_manifest(
    plan: ResolvedRunPlan, spec: BatchGatewaySpec
) -> dict[str, Any]:
    name = community_component_name(plan, "gc")
    labels = _community_labels(plan, "batch-gateway-gc")
    config = _community_gc_config(plan, spec)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": plan.deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": _community_selector(plan, "batch-gateway-gc")},
            "template": {
                "metadata": {
                    "annotations": {
                        "checksum/config": hashlib.sha256(
                            yaml.safe_dump(config, sort_keys=False).encode("utf-8")
                        ).hexdigest(),
                    },
                    "labels": _community_labels(plan, "batch-gateway-gc"),
                },
                "spec": {
                    "serviceAccountName": name,
                    "securityContext": _community_pod_security_context(),
                    "containers": [
                        {
                            "name": "gc",
                            "image": spec.images["gc"],
                            "imagePullPolicy": "IfNotPresent",
                            "securityContext": _community_container_security_context(),
                            "args": [
                                "--config=/etc/batch-gc/config.yaml",
                                "--v=2",
                            ],
                            "ports": [{"name": "metrics", "containerPort": 9091}],
                            "startupProbe": {
                                "httpGet": {"path": "/health", "port": "metrics"},
                                "periodSeconds": 5,
                                "failureThreshold": 60,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": "metrics"},
                                "periodSeconds": 30,
                            },
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "128Mi"},
                                "requests": {"cpu": "10m", "memory": "32Mi"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "config",
                                    "mountPath": "/etc/batch-gc",
                                    "readOnly": True,
                                },
                                *_community_files_mounts(spec),
                                *_community_secret_mounts(),
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "config", "configMap": {"name": f"{name}-config"}},
                        *_community_files_volumes(spec),
                        *_community_secret_volumes(spec),
                    ],
                },
            },
        },
    }


def _community_async_transport_config(
    spec: BatchGatewaySpec,
    *,
    pool_name: str,
    model_url: str,
) -> str:
    config: dict[str, Any] = {
        "result_queue_name": f"llm-d-async:results:{pool_name}",
        "queues": [
            {
                "queue_name": f"llm-d-async:requests:{pool_name}",
                "igw_base_url": model_url,
                "request_path_url": "/v1/chat/completions",
                "inference_objective": spec.inference_objective,
            }
        ],
    }
    if spec.transport == "redis-sortedset":
        config = {
            "batch_size": 10,
            "poll_interval_ms": 500,
            **config,
        }
    # The connection URL is injected through the SQL_URL / REDIS_URL
    # environment variable so it never lands in rendered manifests.
    return json.dumps(config)


def render_community_llm_d_async_manifest(
    plan: ResolvedRunPlan,
    spec: BatchGatewaySpec,
    *,
    pool_name: str,
    model_url: str,
) -> dict[str, Any]:
    name = llm_d_async_name(plan)
    url_env = "SQL_URL" if spec.transport == "sql" else "REDIS_URL"
    url_key = "postgresql-url" if spec.transport == "sql" else "redis-url"
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": plan.deployment.namespace,
            "labels": _community_labels(plan, "llm-d-async"),
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": _community_selector(plan, "llm-d-async")},
            "template": {
                "metadata": {
                    "labels": _community_labels(plan, "llm-d-async"),
                },
                "spec": {
                    "serviceAccountName": name,
                    "securityContext": _community_pod_security_context(),
                    "containers": [
                        {
                            "name": "llm-d-async",
                            "image": spec.images["async"],
                            "imagePullPolicy": "Always",
                            "securityContext": _community_container_security_context(),
                            "command": ["/llm-d-async"],
                            "args": [
                                f"--transport={spec.transport}",
                                "--transport-config",
                                _community_async_transport_config(
                                    spec, pool_name=pool_name, model_url=model_url
                                ),
                                "--concurrency=256",
                                "--gate-wait-timeout=5m",
                                "--drain-timeout=2m",
                                "--request-timeout=30m",
                                "--health-port=8081",
                                "--metrics-endpoint-auth=false",
                                "--metrics-port=9090",
                            ],
                            "env": [
                                {"name": "LOG_LEVEL", "value": "info"},
                                {
                                    "name": url_env,
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": spec.secret_name,
                                            "key": url_key,
                                        }
                                    },
                                },
                            ],
                            "ports": [
                                {"name": "metrics", "containerPort": 9090},
                                {"name": "health", "containerPort": 8081},
                            ],
                            "startupProbe": {
                                "httpGet": {"path": "/healthz", "port": "health"},
                                "periodSeconds": 2,
                                "timeoutSeconds": 3,
                                "failureThreshold": 30,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/readyz", "port": "health"},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": "health"},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3,
                            },
                            "resources": {
                                "limits": {"cpu": "1", "memory": "1Gi"},
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                            },
                        }
                    ],
                },
            },
        },
    }


def render_community_batch_gateway_manifests(
    plan: ResolvedRunPlan,
    *,
    spec: BatchGatewaySpec,
    pool_name: str,
    model_url: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Render every community batch gateway resource for one release.

    Returns (filename, manifest) pairs in apply order. All resources carry
    the benchflow release labels so cleanup deletes them with the same
    selector as the operator path.
    """
    namespace = plan.deployment.namespace
    apiserver_name = community_component_name(plan, "apiserver")
    processor_name = community_component_name(plan, "processor")
    gc_name = community_component_name(plan, "gc")
    async_name = llm_d_async_name(plan)

    service_accounts = [
        _community_service_account(
            component_name,
            _community_labels(plan, component),
            namespace,
        )
        for component_name, component in (
            (apiserver_name, "batch-gateway-apiserver"),
            (processor_name, "batch-gateway-processor"),
            (gc_name, "batch-gateway-gc"),
            (async_name, "llm-d-async"),
        )
    ]
    config_maps = [
        _community_config_configmap(
            f"{apiserver_name}-config",
            _community_labels(plan, "batch-gateway-apiserver"),
            namespace,
            _community_apiserver_config(spec),
        ),
        _community_config_configmap(
            f"{processor_name}-config",
            _community_labels(plan, "batch-gateway-processor"),
            namespace,
            _community_processor_config(plan, spec, pool_name=pool_name),
        ),
        _community_config_configmap(
            f"{gc_name}-config",
            _community_labels(plan, "batch-gateway-gc"),
            namespace,
            _community_gc_config(plan, spec),
        ),
    ]
    gc_role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {
            "name": f"{gc_name}-pod-watcher",
            "namespace": namespace,
            "labels": _community_labels(plan, "batch-gateway-gc"),
        },
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["pods"],
                "verbs": ["get", "list", "watch"],
            },
            {
                "apiGroups": ["apps"],
                "resources": ["statefulsets"],
                "verbs": ["get", "list", "watch"],
            },
        ],
    }
    gc_role_binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {
            "name": f"{gc_name}-pod-watcher",
            "namespace": namespace,
            "labels": _community_labels(plan, "batch-gateway-gc"),
        },
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": f"{gc_name}-pod-watcher",
        },
        "subjects": [
            {"kind": "ServiceAccount", "name": gc_name, "namespace": namespace}
        ],
    }
    apiserver_service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": apiserver_name,
            "namespace": namespace,
            "labels": _community_labels(plan, "batch-gateway-apiserver"),
        },
        "spec": {
            "type": "ClusterIP",
            "ports": [
                {"port": 8000, "targetPort": "http", "name": "http"},
                {"port": 8081, "targetPort": "observability", "name": "observability"},
            ],
            "selector": _community_selector(plan, "batch-gateway-apiserver"),
        },
    }
    processor_service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": processor_name,
            "namespace": namespace,
            "labels": _community_labels(plan, "batch-gateway-processor"),
        },
        "spec": {
            "clusterIP": "None",
            "selector": _community_selector(plan, "batch-gateway-processor"),
        },
    }
    workloads = [
        render_community_apiserver_manifest(plan, spec),
        render_community_processor_manifest(plan, spec, pool_name=pool_name),
        render_community_gc_manifest(plan, spec),
        render_community_llm_d_async_manifest(
            plan, spec, pool_name=pool_name, model_url=model_url
        ),
    ]
    manifests: list[tuple[str, dict[str, Any]]] = []
    for name, manifest in zip(
        (
            "batch-apiserver-serviceaccount.yaml",
            "batch-processor-serviceaccount.yaml",
            "batch-gc-serviceaccount.yaml",
            "batch-async-serviceaccount.yaml",
        ),
        service_accounts,
        strict=True,
    ):
        manifests.append((name, manifest))
    for name, manifest in zip(
        (
            "batch-apiserver-configmap.yaml",
            "batch-processor-configmap.yaml",
            "batch-gc-configmap.yaml",
        ),
        config_maps,
        strict=True,
    ):
        manifests.append((name, manifest))
    manifests.append(("batch-gc-role.yaml", gc_role))
    manifests.append(("batch-gc-rolebinding.yaml", gc_role_binding))
    for name, manifest in zip(
        (
            "batch-apiserver-deployment.yaml",
            "batch-processor-statefulset.yaml",
            "batch-gc-deployment.yaml",
            "batch-async-deployment.yaml",
        ),
        workloads,
        strict=True,
    ):
        manifests.append((name, manifest))
    manifests.append(("batch-apiserver-service.yaml", apiserver_service))
    manifests.append(("batch-processor-service.yaml", processor_service))
    return manifests


def wait_for_community_batch_gateway_ready(
    kubectl_cmd: str,
    plan: ResolvedRunPlan,
    spec: BatchGatewaySpec,
    *,
    timeout_seconds: int,
) -> None:
    targets = [
        ("deployment", community_component_name(plan, "apiserver")),
        ("statefulset", community_component_name(plan, "processor")),
        ("deployment", community_component_name(plan, "gc")),
        ("deployment", llm_d_async_name(plan)),
    ]
    namespace = plan.deployment.namespace
    pending = list(targets)
    deadline = time.time() + timeout_seconds
    reported: set[tuple[str, str]] = set()
    while time.time() < deadline and pending:
        still_pending: list[tuple[str, str]] = []
        for kind, name in pending:
            result = run_command(
                [kubectl_cmd, "get", kind, name, "-n", namespace, "-o", "json"],
                capture_output=True,
                check=False,
            )
            ready = False
            if result.returncode == 0:
                try:
                    payload = json.loads(result.stdout or "{}")
                except json.JSONDecodeError:
                    payload = {}
                status = payload.get("status") or {}
                ready = int(status.get("readyReplicas") or 0) >= 1 and not (
                    payload.get("spec", {}).get("replicas", 1) > 0
                    and int(status.get("replicas") or 0) == 0
                )
            if not ready:
                still_pending.append((kind, name))
                if (kind, name) not in reported:
                    detail(f"Waiting for community batch gateway {kind} {name}")
                    reported.add((kind, name))
        pending = still_pending
        if pending:
            time.sleep(10)
    if pending:
        raise CommandError(
            "timed out waiting for the community batch gateway: "
            + ", ".join(f"{kind}/{name}" for kind, name in pending)
        )


def render_batch_route_manifests(
    plan: ResolvedRunPlan,
    *,
    spec: BatchGatewaySpec,
    apiserver_service: str,
) -> list[dict[str, Any]]:
    """External /v1/batches + /v1/files route on the release Gateway."""
    namespace = plan.deployment.namespace
    manifests: list[dict[str, Any]] = []
    if spec.tls_enabled:
        manifests.append(
            {
                "apiVersion": "networking.istio.io/v1",
                "kind": "DestinationRule",
                "metadata": {
                    "name": batch_destinationrule_name(plan),
                    "namespace": RHOAI_GATEWAY_NAMESPACE,
                    "labels": _batch_labels(plan, "batch-gateway-tls"),
                },
                "spec": {
                    "host": (f"{apiserver_service}.{namespace}.svc.cluster.local"),
                    "trafficPolicy": {
                        "portLevelSettings": [
                            {
                                "port": {"number": 8000},
                                "tls": {
                                    "mode": "SIMPLE",
                                    "insecureSkipVerify": True,
                                },
                            }
                        ]
                    },
                },
            }
        )
    manifests.append(
        {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "HTTPRoute",
            "metadata": {
                "name": batch_route_name(plan),
                "namespace": namespace,
                "labels": _batch_labels(plan, "batch-route"),
            },
            "spec": {
                "parentRefs": [
                    {
                        "name": rhoai_release_gateway_name(plan),
                        "namespace": RHOAI_GATEWAY_NAMESPACE,
                    }
                ],
                "rules": [
                    {
                        "matches": [
                            {
                                "path": {
                                    "type": "PathPrefix",
                                    "value": "/v1/batches",
                                }
                            },
                            {
                                "path": {
                                    "type": "PathPrefix",
                                    "value": "/v1/files",
                                }
                            },
                        ],
                        "backendRefs": [{"name": apiserver_service, "port": 8000}],
                    }
                ],
            },
        }
    )
    manifests.append(
        {
            "apiVersion": "kuadrant.io/v1",
            "kind": "AuthPolicy",
            "metadata": {
                "name": batch_route_authpolicy_name(plan),
                "namespace": namespace,
                "labels": _batch_labels(plan, "batch-route-auth"),
            },
            "spec": {
                "targetRef": {
                    "group": "gateway.networking.k8s.io",
                    "kind": "HTTPRoute",
                    "name": batch_route_name(plan),
                },
                "rules": {
                    "authentication": {
                        "kubernetes-user": {
                            "kubernetesTokenReview": {
                                "audiences": ["https://kubernetes.default.svc"]
                            }
                        }
                    }
                },
            },
        }
    )
    return manifests


def render_batch_files_pvc_manifest(
    plan: ResolvedRunPlan,
    *,
    claim_name: str,
    storage_class_name: str = "",
) -> dict[str, Any]:
    manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": claim_name,
            "namespace": plan.deployment.namespace,
            "labels": _batch_labels(plan, "batch-files"),
        },
        "spec": {
            "accessModes": list(BATCH_FILES_PVC_ACCESS_MODES),
            "resources": {"requests": {"storage": BATCH_FILES_PVC_SIZE}},
        },
    }
    if storage_class_name:
        manifest["spec"]["storageClassName"] = storage_class_name
    return manifest


# ---------------------------------------------------------------------------
# Deploy and cleanup orchestration


def _resource_type_unavailable(stderr: str) -> bool:
    return (
        "the server doesn't have a resource type" in stderr
        or "the server does not have a resource type" in stderr
        or "no matches for kind" in stderr
    )


def _apply_manifest(
    kubectl_cmd: str,
    manifest: dict[str, Any],
    *,
    graceful_missing_kind: bool = False,
) -> bool:
    kind = str(manifest.get("kind") or "resource")
    name = str((manifest.get("metadata") or {}).get("name") or "")
    result = run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    stderr = str(result.stderr or "")
    if graceful_missing_kind and _resource_type_unavailable(stderr):
        detail(
            f"Skipping {kind} {name}: the Kuadrant APIs are not installed on this "
            "cluster; the batch gateway runs without gateway-level authentication"
        )
        return False
    raise CommandError(
        f"failed to apply {kind} {name}: {stderr or str(result.stdout or '').strip()}"
    )


def _write_manifest(
    manifests_dir: Path | None, filename: str, manifest: dict[str, Any]
) -> None:
    if manifests_dir is None:
        return
    manifests_dir.mkdir(parents=True, exist_ok=True)
    target = manifests_dir / filename
    target.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    detail(f"Rendered batch gateway manifest written to {target}")


def _require_secret(
    kubectl_cmd: str,
    namespace: str,
    secret_name: str,
    *,
    required_keys: tuple[str, ...] = (),
) -> None:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "secret",
            secret_name,
            "-n",
            namespace,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CommandError(
            f"Secret {secret_name} is required in namespace {namespace} before the "
            "batch gateway can be deployed. Create it with the redis-url and "
            "postgresql-url keys (see docs/RHOAI_BATCH_GATEWAY.md and the upstream "
            f"guide {_UPSTREAM_GUIDE_URL}, section 3.9)"
        )
    if not required_keys:
        return
    try:
        payload = json.loads(result.stdout or "{}")
        present = set((payload.get("data") or {}).keys())
    except json.JSONDecodeError:
        present = set()
    missing = [key for key in required_keys if key not in present]
    if missing:
        raise CommandError(
            f"Secret {secret_name} in namespace {namespace} is missing the "
            + ", ".join(missing)
            + " key(s) required by the configured batch gateway transport"
        )


def _ensure_namespace_gateway_label(kubectl_cmd: str, namespace: str) -> None:
    run_command(
        [
            kubectl_cmd,
            "label",
            "namespace",
            namespace,
            f"{GATEWAY_ROUTE_NAMESPACE_LABEL}=true",
            "--overwrite",
        ]
    )


def _ensure_file_storage_pvc(
    plan: ResolvedRunPlan,
    spec: BatchGatewaySpec,
    *,
    kubectl_cmd: str,
    manifests_dir: Path | None,
) -> None:
    if not spec.uses_fs_storage:
        return
    claim_name = spec.fs_claim_name
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "pvc",
            claim_name,
            "-n",
            plan.deployment.namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        detail(f"Reusing batch files PVC {claim_name}")
        return
    manifest = render_batch_files_pvc_manifest(
        plan,
        claim_name=claim_name,
        storage_class_name=spec.fs_storage_class_name,
    )
    _apply_manifest(kubectl_cmd, manifest)
    _write_manifest(manifests_dir, "batch-files-pvc.yaml", manifest)
    storage_class_detail = (
        f", storageClassName {spec.fs_storage_class_name}"
        if spec.fs_storage_class_name
        else ""
    )
    detail(
        f"Created batch files PVC {claim_name} "
        f"({BATCH_FILES_PVC_SIZE} {','.join(BATCH_FILES_PVC_ACCESS_MODES)}"
        f"{storage_class_detail})"
    )


def deploy_rhoai_batch_gateway(
    plan: ResolvedRunPlan,
    *,
    kubectl_cmd: str,
    manifests_dir: Path | None = None,
    timeout_seconds: int = 1800,
) -> None:
    """Deploy the operator-managed batch gateway for one release.

    Requires the batch gateway operator (DSC aigateway.batchGateway) to be
    installed by setup, and the release LLMInferenceService to be applied.
    """
    spec = rhoai_batch_gateway_spec(plan)
    if spec is None:
        return
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name

    step("Validating batch gateway prerequisites")
    secret_keys: tuple[str, ...] = ("postgresql-url",)
    if spec.transport == "redis-sortedset" and spec.community_implementation:
        secret_keys = ("postgresql-url", "redis-url")
    _require_secret(
        kubectl_cmd, namespace, spec.secret_name, required_keys=secret_keys
    )
    if spec.community_implementation:
        api_version = ""
        # The community llm-d-router EPP prefers the llm-d.ai InferenceObjective
        # CRD when both groups are installed; mirror its discovery order.
        objective_api_version = community_inference_objective_api_version(kubectl_cmd)
        detail("LLMBatchGateway API: (community components)")
    else:
        api_version = llmbatchgateway_api_version(kubectl_cmd)
        objective_api_version = inference_objective_api_version(kubectl_cmd)
        detail(f"LLMBatchGateway API: {api_version}")
    detail(f"InferenceObjective API: {objective_api_version}")
    pool_group = inference_pool_group(kubectl_cmd)

    step("Preparing the batch internal Gateway")
    _ensure_namespace_gateway_label(kubectl_cmd, namespace)
    gateway_config = load_rhoai_gateway_configuration(kubectl_cmd)
    internal_gateway = render_batch_internal_gateway_manifest(plan, gateway_config)
    _apply_manifest(kubectl_cmd, internal_gateway)
    _write_manifest(manifests_dir, "batch-internal-gateway.yaml", internal_gateway)
    internal_gateway_name = rhoai_batch_internal_gateway_name(plan)
    wait_for_rhoai_gateway_ready(
        kubectl_cmd,
        namespace=RHOAI_GATEWAY_NAMESPACE,
        name=internal_gateway_name,
        timeout_seconds=min(timeout_seconds, 900),
    )
    internal_service = discover_batch_internal_gateway_service(
        kubectl_cmd,
        gateway_name=internal_gateway_name,
        timeout_seconds=600,
    )
    model_url = (
        f"http://{internal_service}.{RHOAI_GATEWAY_NAMESPACE}.svc.cluster.local"
        f"/{namespace}/{release_name}"
    )
    detail(f"Batch model URL: {model_url}")

    step("Routing batch inference through the internal Gateway")
    pool_name = discover_inference_pool_name(
        kubectl_cmd,
        namespace=namespace,
        isvc_name=release_name,
        timeout_seconds=min(timeout_seconds, 900),
    )
    detail(f"InferencePool: {namespace}/{pool_name}")
    _ensure_file_storage_pvc(
        plan, spec, kubectl_cmd=kubectl_cmd, manifests_dir=manifests_dir
    )
    objective = render_batch_inference_objective_manifest(
        plan,
        spec=spec,
        api_version=objective_api_version,
        pool_group=pool_group,
        pool_name=pool_name,
    )
    _apply_manifest(kubectl_cmd, objective)
    _write_manifest(manifests_dir, "batch-inference-objective.yaml", objective)
    if spec.interactive_objective:
        interactive_objective = render_batch_inference_objective_manifest(
            plan,
            spec=spec,
            api_version=objective_api_version,
            pool_group=pool_group,
            pool_name=pool_name,
            name=spec.interactive_objective,
            priority=0,
            component="interactive-inference-objective",
        )
        _apply_manifest(kubectl_cmd, interactive_objective)
        _write_manifest(
            manifests_dir, "interactive-inference-objective.yaml", interactive_objective
        )
    llm_route = render_batch_llm_route_manifest(
        plan, pool_group=pool_group, pool_name=pool_name
    )
    _apply_manifest(kubectl_cmd, llm_route)
    _write_manifest(manifests_dir, "batch-llm-route.yaml", llm_route)
    if spec.internal_route_auth:
        llm_route_auth = render_batch_llm_route_authpolicy(plan)
        _apply_manifest(kubectl_cmd, llm_route_auth, graceful_missing_kind=True)
        _write_manifest(
            manifests_dir, "batch-llm-route-authpolicy.yaml", llm_route_auth
        )
    else:
        detail(
            "Internal route authentication is disabled for this release "
            "(options.batch_gateway.internal_route_auth: false)"
        )

    if spec.community_implementation:
        step("Applying the community batch gateway components")
        community_manifests = render_community_batch_gateway_manifests(
            plan,
            spec=spec,
            pool_name=pool_name,
            model_url=model_url,
        )
        for filename, manifest in community_manifests:
            _apply_manifest(kubectl_cmd, manifest)
            _write_manifest(manifests_dir, filename, manifest)
        wait_for_community_batch_gateway_ready(
            kubectl_cmd,
            plan,
            spec,
            timeout_seconds=timeout_seconds,
        )
        success(
            "Community batch gateway components are ready "
            f"(transport: {spec.transport})"
        )
        apiserver_service = community_component_name(plan, "apiserver")
    else:
        step(f"Applying LLMBatchGateway {llmbatchgateway_name(plan)}")
        batch_gateway = render_llmbatchgateway_manifest(
            plan,
            spec=spec,
            api_version=api_version,
            pool_name=pool_name,
            model_url=model_url,
        )
        _apply_manifest(kubectl_cmd, batch_gateway)
        _write_manifest(manifests_dir, "llmbatchgateway.yaml", batch_gateway)
        wait_for_llmbatchgateway_ready(
            kubectl_cmd,
            namespace=namespace,
            name=llmbatchgateway_name(plan),
            timeout_seconds=timeout_seconds,
        )
        success(f"LLMBatchGateway {llmbatchgateway_name(plan)} is ready")
        apiserver_service = discover_batch_apiserver_service(
            kubectl_cmd,
            namespace=namespace,
            cr_name=llmbatchgateway_name(plan),
            timeout_seconds=600,
        )

    step("Exposing the batch API on the release Gateway")
    detail(f"Batch apiserver Service: {namespace}/{apiserver_service}")
    route_manifests = render_batch_route_manifests(
        plan,
        spec=spec,
        apiserver_service=apiserver_service,
    )
    route_filenames = ["batch-route.yaml", "batch-route-authpolicy.yaml"]
    if spec.tls_enabled:
        route_filenames.insert(0, "batch-destinationrule.yaml")
    for manifest, filename in zip(route_manifests, route_filenames, strict=True):
        _apply_manifest(
            kubectl_cmd,
            manifest,
            graceful_missing_kind=manifest["kind"] == "AuthPolicy",
        )
        _write_manifest(manifests_dir, filename, manifest)
    endpoint = rhoai_batch_endpoint_url(kubectl_cmd, plan)
    success(f"Batch API is available at {endpoint}/v1/batches")
    detail(
        "Authenticate with a Kubernetes service-account token; BenchFlow does not "
        "drive the /v1/batches API itself yet"
    )


def cleanup_rhoai_batch_gateway(
    plan: ResolvedRunPlan,
    *,
    kubectl_cmd: str,
    timeout_seconds: int = 300,
) -> None:
    """Remove every benchflow-labeled batch gateway resource for one release."""
    if not rhoai_batch_gateway_requested(plan):
        return
    namespace = plan.deployment.namespace
    selector = _batch_release_selector(plan)

    step(f"Removing the LLMBatchGateway for release {plan.deployment.release_name}")
    run_command(
        [
            kubectl_cmd,
            "delete",
            "llmbatchgateway",
            "-n",
            namespace,
            "-l",
            selector,
            "--ignore-not-found=true",
        ],
        check=False,
    )
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "llmbatchgateway",
                "-n",
                namespace,
                "-l",
                selector,
                "-o",
                "name",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not str(result.stdout or "").strip():
            break
        time.sleep(5)
    else:
        detail("LLMBatchGateway deletion still in progress; continuing cleanup")

    for resources, resource_namespace in (
        ("httproute", namespace),
        # Both InferenceObjective CRD groups are cleaned explicitly: the bare
        # resource name resolves to whichever CRD oc picks first, and the
        # community EPP path creates objectives under llm-d.ai while the
        # operator path may use the legacy inference.networking.x-k8s.io group.
        ("inferenceobjectives.llm-d.ai", namespace),
        ("inferenceobjectives.inference.networking.x-k8s.io", namespace),
        ("authpolicy", namespace),
        ("deployment", namespace),
        ("statefulset", namespace),
        ("configmap", namespace),
        ("service", namespace),
        ("serviceaccount", namespace),
        ("role.rbac.authorization.k8s.io", namespace),
        ("rolebinding.rbac.authorization.k8s.io", namespace),
        ("pvc", namespace),
        ("gateway", RHOAI_GATEWAY_NAMESPACE),
        ("destinationrule", RHOAI_GATEWAY_NAMESPACE),
    ):
        run_command(
            [
                kubectl_cmd,
                "delete",
                resources,
                "-n",
                resource_namespace,
                "-l",
                selector,
                "--ignore-not-found=true",
            ],
            check=False,
        )
    detail(
        "Removed batch gateway routes, policies, objectives, workloads "
        "(operator or community components), the internal Gateway, and "
        "benchflow-created batch files PVCs"
    )
