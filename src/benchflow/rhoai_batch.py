from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
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
    "request_timeout",
    "max_retries",
    "initial_backoff",
    "max_backoff",
    "inference_objective",
    "async_config",
    "gc_interval",
    "tls",
}
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
    unknown = sorted(set(fs) - {"claim_name", "base_path"})
    if unknown:
        raise ValidationError(
            "options.batch_gateway.file_storage.fs contains unsupported fields: "
            + ", ".join(unknown)
        )
    return {
        "fs": {
            "claim_name": str(fs.get("claim_name") or "").strip(),
            "base_path": str(fs.get("base_path") or "/tmp/batch-gateway").strip(),
        }
    }


def _parse_async_config(raw: Any, dispatch_mode: str) -> dict[str, Any]:
    if dispatch_mode == "async":
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

    sync_only_set = [key for key in _SYNC_ONLY_OPTION_KEYS if raw.get(key) is not None]
    if dispatch_mode == "async" and sync_only_set:
        raise ValidationError(
            "options.batch_gateway "
            + ", ".join(sync_only_set)
            + " apply only to dispatch_mode: sync"
        )

    file_storage = _parse_file_storage(raw.get("file_storage"))
    async_config = _parse_async_config(raw.get("async_config"), dispatch_mode)
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
) -> dict[str, Any]:
    """Sheddable objective (priority -1) the batch processor requests."""
    return {
        "apiVersion": api_version,
        "kind": "InferenceObjective",
        "metadata": {
            "name": spec.inference_objective,
            "namespace": plan.deployment.namespace,
            "labels": _batch_labels(plan, "batch-inference-objective"),
        },
        "spec": {
            "priority": -1,
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
) -> dict[str, Any]:
    return {
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


def _require_secret(kubectl_cmd: str, namespace: str, secret_name: str) -> None:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "secret",
            secret_name,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return
    raise CommandError(
        f"Secret {secret_name} is required in namespace {namespace} before the "
        "batch gateway can be deployed. Create it with the redis-url and "
        "postgresql-url keys (see docs/RHOAI_BATCH_GATEWAY.md and the upstream "
        f"guide {_UPSTREAM_GUIDE_URL}, section 3.9)"
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
    manifest = render_batch_files_pvc_manifest(plan, claim_name=claim_name)
    _apply_manifest(kubectl_cmd, manifest)
    _write_manifest(manifests_dir, "batch-files-pvc.yaml", manifest)
    detail(
        f"Created batch files PVC {claim_name} "
        f"({BATCH_FILES_PVC_SIZE} {','.join(BATCH_FILES_PVC_ACCESS_MODES)})"
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
    _require_secret(kubectl_cmd, namespace, spec.secret_name)
    api_version = llmbatchgateway_api_version(kubectl_cmd)
    objective_api_version = inference_objective_api_version(kubectl_cmd)
    pool_group = inference_pool_group(kubectl_cmd)
    detail(f"LLMBatchGateway API: {api_version}")

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
    llm_route = render_batch_llm_route_manifest(
        plan, pool_group=pool_group, pool_name=pool_name
    )
    _apply_manifest(kubectl_cmd, llm_route)
    _write_manifest(manifests_dir, "batch-llm-route.yaml", llm_route)
    llm_route_auth = render_batch_llm_route_authpolicy(plan)
    _apply_manifest(kubectl_cmd, llm_route_auth, graceful_missing_kind=True)
    _write_manifest(manifests_dir, "batch-llm-route-authpolicy.yaml", llm_route_auth)

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

    step("Exposing the batch API on the release Gateway")
    apiserver_service = discover_batch_apiserver_service(
        kubectl_cmd,
        namespace=namespace,
        cr_name=llmbatchgateway_name(plan),
        timeout_seconds=600,
    )
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
        ("inferenceobjective", namespace),
        ("authpolicy", namespace),
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
        "Removed batch gateway routes, policies, objectives, the internal "
        "Gateway, and benchflow-created batch files PVCs"
    )
