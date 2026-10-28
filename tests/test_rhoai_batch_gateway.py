from pathlib import Path

import pytest

from benchflow.loaders import ProfileCatalog, load_deployment_profile, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.rhoai_batch import (
    batch_llm_route_name,
    batch_route_name,
    llmbatchgateway_name,
    parse_batch_gateway_options,
    render_batch_internal_gateway_manifest,
    render_batch_llm_route_manifest,
    render_batch_route_manifests,
    render_llmbatchgateway_manifest,
    rhoai_batch_gateway_spec,
)
from benchflow.rhoai_gateway import RhoaiGatewayConfiguration

REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_API_VERSION = "batch.llm-d.ai/v1alpha1"
TEST_POOL_NAME = "test-inference-pool"
TEST_MODEL_URL = (
    "http://test-gw-svc.openshift-ingress.svc.cluster.local/benchflow/test-isvc"
)


def _plans(experiment: str) -> list:
    return resolve_experiment_matrix(
        load_experiment(REPO_ROOT / f"experiments/rhoai/batch/{experiment}.yaml"),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )


def _gateway_config() -> RhoaiGatewayConfiguration:
    return RhoaiGatewayConfiguration(
        namespace="openshift-ingress",
        gateway_class_name="openshift-ai-inference",
        labels={"istio.io/rev": "openshift-gateway"},
        listener={
            "name": "https",
            "port": 443,
            "protocol": "HTTPS",
            "tls": {"mode": "Terminate"},
        },
    )


def test_batch_experiments_resolve_with_batch_profiles() -> None:
    for experiment, profile in (
        ("batch-gateway-sync", "batch-gateway-sync"),
        ("batch-gateway-async", "batch-gateway-async"),
    ):
        plans = _plans(experiment)
        assert len(plans) == 1
        plan = plans[0]
        assert plan.profiles.deployment == profile
        assert plan.deployment.platform == "rhoai"
        assert plan.deployment.mode == "distributed-default"
        assert plan.deployment.platform_version == "RHOAI-3.6.0-ea.2"
        assert plan.deployment.platform_channel == "beta"
        assert "batch_gateway" in plan.deployment.options
        assert plan.stages.benchmark is False


def test_parse_batch_gateway_options_sync_defaults() -> None:
    parsed = parse_batch_gateway_options(
        {
            "secret_name": "batch-gateway-secrets",
            "file_storage": {"fs": {}},
        }
    )
    assert parsed["db_backend"] == "postgresql"
    assert parsed["dispatch_mode"] == "sync"
    assert parsed["api_server_replicas"] == 1
    assert parsed["processor_replicas"] == 1
    assert parsed["request_timeout"] == "5m"
    assert parsed["max_retries"] == 3
    assert parsed["initial_backoff"] == "1s"
    assert parsed["max_backoff"] == "60s"
    assert parsed["gc_interval"] == "30m"
    assert parsed["async_config"] == {}
    assert parsed["tls"] == {}


def test_parse_batch_gateway_options_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="unsupported fields: color"):
        parse_batch_gateway_options(
            {"secret_name": "s", "file_storage": {"fs": {}}, "color": "blue"}
        )


def test_parse_batch_gateway_options_rejects_sync_knobs_with_async() -> None:
    with pytest.raises(ValidationError, match="apply only to dispatch_mode: sync"):
        parse_batch_gateway_options(
            {
                "secret_name": "s",
                "dispatch_mode": "async",
                "file_storage": {"fs": {}},
                "async_config": {"redis": {"queuesConfig": []}},
                "max_retries": 5,
            }
        )


def test_parse_batch_gateway_options_requires_exactly_one_file_storage() -> None:
    with pytest.raises(ValidationError, match="exactly one of fs or s3"):
        parse_batch_gateway_options({"secret_name": "s", "file_storage": {}})
    with pytest.raises(ValidationError, match="exactly one of fs or s3"):
        parse_batch_gateway_options(
            {
                "secret_name": "s",
                "file_storage": {"fs": {}, "s3": {"region": "us-east-1"}},
            }
        )


def test_parse_batch_gateway_options_requires_secret_name() -> None:
    with pytest.raises(ValidationError, match="secret_name is required"):
        parse_batch_gateway_options({"file_storage": {"fs": {}}})


def test_parse_batch_gateway_options_requires_async_config_for_async() -> None:
    with pytest.raises(ValidationError, match="required for dispatch_mode: async"):
        parse_batch_gateway_options(
            {
                "secret_name": "s",
                "dispatch_mode": "async",
                "file_storage": {"fs": {}},
            }
        )


def test_spec_fills_release_scoped_defaults() -> None:
    plan = _plans("batch-gateway-sync")[0]
    spec = rhoai_batch_gateway_spec(plan)
    assert spec is not None
    release = plan.deployment.release_name
    assert spec.inference_objective == f"{release}-batch-sheddable"
    assert spec.fs_claim_name == f"{release}-batch-gateway-files"
    assert spec.fs_base_path == "/tmp/batch-gateway"
    assert spec.secret_name == "batch-gateway-secrets"


def test_render_llmbatchgateway_sync_manifest() -> None:
    plan = _plans("batch-gateway-sync")[0]
    spec = rhoai_batch_gateway_spec(plan)
    manifest = render_llmbatchgateway_manifest(
        plan,
        spec=spec,
        api_version=TEST_API_VERSION,
        pool_name=TEST_POOL_NAME,
        model_url=TEST_MODEL_URL,
    )
    assert manifest["apiVersion"] == TEST_API_VERSION
    assert manifest["kind"] == "LLMBatchGateway"
    assert manifest["metadata"]["name"] == llmbatchgateway_name(plan)
    assert manifest["metadata"]["namespace"] == plan.deployment.namespace
    labels = manifest["metadata"]["labels"]
    assert labels["benchflow.io/batch-gateway"] == "true"
    assert labels["benchflow.io/release"] == plan.deployment.release_name

    processor = manifest["spec"]["processor"]
    assert "dispatchMode" not in processor
    model_gateway = processor["modelGateways"][plan.model.name]
    assert model_gateway["url"] == TEST_MODEL_URL
    assert model_gateway["requestTimeout"] == "5m"
    assert model_gateway["maxRetries"] == 3
    assert model_gateway["initialBackoff"] == "1s"
    assert model_gateway["maxBackoff"] == "60s"
    assert model_gateway["inferenceObjective"] == spec.inference_objective
    assert manifest["spec"]["secretRef"] == {"name": "batch-gateway-secrets"}
    assert manifest["spec"]["dbBackend"] == "postgresql"
    assert manifest["spec"]["fileStorage"]["fs"] == {
        "basePath": "/tmp/batch-gateway",
        "claimName": spec.fs_claim_name,
    }
    assert manifest["spec"]["gc"] == {"interval": "30m"}
    assert "tls" not in manifest["spec"]


def test_render_llmbatchgateway_async_manifest_injects_release_defaults() -> None:
    plan = _plans("batch-gateway-async")[0]
    spec = rhoai_batch_gateway_spec(plan)
    assert spec.dispatch_mode == "async"
    manifest = render_llmbatchgateway_manifest(
        plan,
        spec=spec,
        api_version=TEST_API_VERSION,
        pool_name=TEST_POOL_NAME,
        model_url=TEST_MODEL_URL,
    )
    processor = manifest["spec"]["processor"]
    assert processor["dispatchMode"] == "async"
    model_gateway = processor["modelGateways"][plan.model.name]
    assert model_gateway["inferencePoolName"] == TEST_POOL_NAME
    assert "url" not in model_gateway
    assert "requestTimeout" not in model_gateway

    queue = processor["asyncConfig"]["redis"]["queuesConfig"][0]
    assert queue["name"] == f"llm-d-async:requests:{TEST_POOL_NAME}"
    assert queue["igwBaseURL"] == TEST_MODEL_URL
    # local-max-concurrency gates keep their params untouched.
    assert queue["gateParams"] == {"limit": "256"}


def test_async_prometheus_budget_gate_gets_pool_injection() -> None:
    plan = _plans("batch-gateway-async")[0]
    queue = plan.deployment.options["batch_gateway"]["async_config"]["redis"][
        "queuesConfig"
    ][0]
    queue["gateType"] = "prometheus-budget"
    queue.pop("name", None)

    spec = rhoai_batch_gateway_spec(plan)
    manifest = render_llmbatchgateway_manifest(
        plan,
        spec=spec,
        api_version=TEST_API_VERSION,
        pool_name=TEST_POOL_NAME,
        model_url=TEST_MODEL_URL,
    )
    rendered = manifest["spec"]["processor"]["asyncConfig"]["redis"]["queuesConfig"][0]
    assert rendered["gateType"] == "prometheus-budget"
    assert rendered["gateParams"]["pool"] == TEST_POOL_NAME
    assert rendered["gateParams"]["namespace"] == plan.deployment.namespace


def test_render_batch_internal_gateway_manifest() -> None:
    plan = _plans("batch-gateway-sync")[0]
    manifest = render_batch_internal_gateway_manifest(plan, _gateway_config())
    assert manifest["kind"] == "Gateway"
    assert manifest["metadata"]["namespace"] == "openshift-ingress"
    assert (
        manifest["metadata"]["annotations"]["networking.istio.io/service-type"]
        == "ClusterIP"
    )
    assert manifest["metadata"]["labels"]["istio.io/rev"] == "openshift-gateway"
    assert manifest["spec"]["gatewayClassName"] == "openshift-ai-inference"
    listener = manifest["spec"]["listeners"][0]
    assert listener["protocol"] == "HTTP"
    assert listener["allowedRoutes"]["namespaces"]["selector"]["matchLabels"] == {
        "llm-d.ai/gateway-route": "true"
    }


def test_render_batch_llm_route_manifest() -> None:
    plan = _plans("batch-gateway-sync")[0]
    manifest = render_batch_llm_route_manifest(
        plan, pool_group="inference.networking.k8s.io", pool_name=TEST_POOL_NAME
    )
    assert manifest["kind"] == "HTTPRoute"
    assert manifest["metadata"]["name"] == batch_llm_route_name(plan)
    assert manifest["metadata"]["namespace"] == plan.deployment.namespace
    assert manifest["spec"]["parentRefs"][0]["namespace"] == "openshift-ingress"
    namespace = plan.deployment.namespace
    isvc = plan.deployment.release_name
    prefixes = {
        rule["matches"][0]["path"]["value"] for rule in manifest["spec"]["rules"]
    }
    assert prefixes == {
        f"/{namespace}/{isvc}/v1/completions",
        f"/{namespace}/{isvc}/v1/chat/completions",
    }
    for rule in manifest["spec"]["rules"]:
        rewrite = rule["filters"][0]["urlRewrite"]["path"]
        assert rewrite["type"] == "ReplacePrefixMatch"
        assert rewrite["replacePrefixMatch"].startswith("/v1/")
        backend = rule["backendRefs"][0]
        assert backend["kind"] == "InferencePool"
        assert backend["name"] == TEST_POOL_NAME


def test_render_batch_route_manifests() -> None:
    plan = _plans("batch-gateway-sync")[0]
    spec = rhoai_batch_gateway_spec(plan)
    manifests = render_batch_route_manifests(
        plan, spec=spec, apiserver_service="test-apiserver"
    )
    assert [item["kind"] for item in manifests] == ["HTTPRoute", "AuthPolicy"]
    route, authpolicy = manifests
    assert route["metadata"]["name"] == batch_route_name(plan)
    assert route["spec"]["parentRefs"][0]["namespace"] == "openshift-ingress"
    paths = {
        match["path"]["value"]
        for rule in route["spec"]["rules"]
        for match in rule["matches"]
    }
    assert paths == {"/v1/batches", "/v1/files"}
    assert route["spec"]["rules"][0]["backendRefs"] == [
        {"name": "test-apiserver", "port": 8000}
    ]
    assert authpolicy["spec"]["targetRef"]["kind"] == "HTTPRoute"
    assert authpolicy["spec"]["targetRef"]["name"] == batch_route_name(plan)


def _write_profile(tmp_path: Path, spec_body: str) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "apiVersion: benchflow.io/v1alpha1\n"
        "kind: DeploymentProfile\n"
        "metadata:\n"
        "  name: tmp-batch-profile\n"
        f"spec:\n{spec_body}",
        encoding="utf-8",
    )
    return path


def test_loader_rejects_batch_gateway_on_non_rhoai_platform(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        "  platform: llm-d\n"
        "  mode: distributed-default\n"
        "  options:\n"
        "    batch_gateway:\n"
        "      secret_name: s\n",
    )
    with pytest.raises(ValidationError, match="only for rhoai"):
        load_deployment_profile(path)


def test_loader_rejects_batch_gateway_on_isvc_mode(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        "  platform: rhoai\n"
        "  mode: isvc\n"
        "  options:\n"
        "    batch_gateway:\n"
        "      secret_name: s\n",
    )
    with pytest.raises(ValidationError, match="distributed-default"):
        load_deployment_profile(path)


def test_loader_validates_batch_gateway_options(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        "  platform: rhoai\n"
        "  mode: distributed-default\n"
        "  options:\n"
        "    batch_gateway:\n"
        "      secret_name: s\n"
        "      dispatch_mode: telnet\n",
    )
    with pytest.raises(ValidationError, match="dispatch_mode"):
        load_deployment_profile(path)
