import json
from pathlib import Path

import pytest
import yaml

from benchflow.loaders import ProfileCatalog, load_deployment_profile, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.rhoai_batch import (
    batch_llm_route_authpolicy_name,
    batch_llm_route_name,
    batch_route_name,
    community_component_name,
    deploy_rhoai_batch_gateway,
    llm_d_async_name,
    llmbatchgateway_name,
    parse_batch_gateway_options,
    render_batch_files_pvc_manifest,
    render_batch_inference_objective_manifest,
    render_batch_internal_gateway_manifest,
    render_batch_llm_route_manifest,
    render_batch_route_manifests,
    render_community_batch_gateway_manifests,
    render_community_llm_d_async_manifest,
    render_llmbatchgateway_manifest,
    rhoai_batch_endpoint_url,
    rhoai_batch_gateway_spec,
)
from benchflow.rhoai_gateway import (
    RhoaiGatewayConfiguration,
    rhoai_release_gateway_name,
)

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
        assert plan.deployment.platform_version == "RHOAI-3.5.0"
        assert plan.deployment.platform_channel == "stable-3.5"
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


def test_parse_file_storage_fs_storage_class_name() -> None:
    parsed = parse_batch_gateway_options(
        {
            "secret_name": "s",
            "file_storage": {"fs": {"storage_class_name": "rook-cephfs-fast"}},
        }
    )
    assert parsed["file_storage"]["fs"]["storage_class_name"] == "rook-cephfs-fast"
    defaulted = parse_batch_gateway_options(
        {"secret_name": "s", "file_storage": {"fs": {}}}
    )
    assert defaulted["file_storage"]["fs"]["storage_class_name"] == ""
    with pytest.raises(ValidationError, match="unsupported fields: zone"):
        parse_batch_gateway_options(
            {"secret_name": "s", "file_storage": {"fs": {"zone": "a"}}}
        )


def test_render_batch_files_pvc_manifest_storage_class() -> None:
    plan = _plans("batch-gateway-sync")[0]
    manifest = render_batch_files_pvc_manifest(
        plan, claim_name="files", storage_class_name="rook-cephfs-fast"
    )
    assert manifest["kind"] == "PersistentVolumeClaim"
    assert manifest["spec"]["accessModes"] == ["ReadWriteMany"]
    assert manifest["spec"]["storageClassName"] == "rook-cephfs-fast"
    defaulted = render_batch_files_pvc_manifest(plan, claim_name="files")
    assert "storageClassName" not in defaulted["spec"]


def test_rhoai_batch_endpoint_url_prefers_listener_hostname(monkeypatch) -> None:
    plan = _plans("batch-gateway-sync")[0]

    def fake_run_json_command(argv):
        assert argv[1:4] == ["get", "gateway", rhoai_release_gateway_name(plan)]
        return {
            "spec": {"listeners": [{"hostname": "gw.example.apps.cluster.com"}]},
            "status": {"addresses": [{"value": "raw-lb.example.com"}]},
        }

    monkeypatch.setattr("benchflow.rhoai_batch.run_json_command", fake_run_json_command)
    assert rhoai_batch_endpoint_url("oc", plan) == "https://gw.example.apps.cluster.com"


def test_rhoai_batch_endpoint_url_falls_back_to_status_address(monkeypatch) -> None:
    plan = _plans("batch-gateway-sync")[0]

    def fake_run_json_command(argv):
        return {
            "spec": {"listeners": [{"hostname": ""}]},
            "status": {"addresses": [{"value": "http://raw-lb.example.com"}]},
        }

    monkeypatch.setattr("benchflow.rhoai_batch.run_json_command", fake_run_json_command)
    assert rhoai_batch_endpoint_url("oc", plan) == "http://raw-lb.example.com"


def test_spec_fills_release_scoped_defaults() -> None:
    plan = _plans("batch-gateway-sync")[0]
    spec = rhoai_batch_gateway_spec(plan)
    assert spec is not None
    release = plan.deployment.release_name
    assert spec.inference_objective == f"{release}-batch-sheddable"
    assert spec.fs_claim_name == f"{release}-batch-gateway-files"
    assert spec.fs_base_path == "/tmp/batch-gateway"
    assert spec.fs_storage_class_name == "rook-cephfs-fast"
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


# ---------------------------------------------------------------------------
# Community implementation + SQL transport


COMMUNITY_IMAGES = {
    "apiserver": "ghcr.io/llm-d/batch-gateway-apiserver@sha256:aaa",
    "processor": "ghcr.io/llm-d/batch-gateway-processor@sha256:bbb",
    "gc": "ghcr.io/llm-d/batch-gateway-gc@sha256:ccc",
    "async": "ghcr.io/llm-d/llm-d-async@sha256:ddd",
}


def _community_options() -> dict:
    return {
        "secret_name": "batch-gateway-sql-secrets",
        "db_backend": "postgresql",
        "dispatch_mode": "async",
        "implementation": "community",
        "transport": "sql",
        "images": dict(COMMUNITY_IMAGES),
        "inference_objective": "batch-low",
        "interactive_objective": "interactive-high",
        "internal_route_auth": False,
        "file_storage": {"fs": {"base_path": "/tmp/batch-gateway"}},
    }


def test_parse_batch_gateway_options_community_sql_defaults() -> None:
    parsed = parse_batch_gateway_options(_community_options())
    assert parsed["implementation"] == "community"
    assert parsed["transport"] == "sql"
    assert parsed["images"] == COMMUNITY_IMAGES
    assert parsed["interactive_objective"] == "interactive-high"
    assert parsed["internal_route_auth"] is False
    # async_config is BenchFlow-derived for the community implementation.
    assert parsed["async_config"] == {}
    # Defaults on the untouched path stay operator + redis.
    defaults = parse_batch_gateway_options({"secret_name": "s", "file_storage": {"fs": {}}})
    assert defaults["implementation"] == "operator"
    assert defaults["transport"] == "redis-sortedset"
    assert defaults["internal_route_auth"] is True
    assert defaults["interactive_objective"] == ""


def test_parse_batch_gateway_options_rejects_sql_without_community() -> None:
    options = _community_options()
    options["implementation"] = "operator"
    with pytest.raises(ValidationError, match="sql requires implementation"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_rejects_sql_with_sync_dispatch() -> None:
    options = _community_options()
    options["dispatch_mode"] = "sync"
    with pytest.raises(ValidationError, match="sql requires dispatch_mode"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_rejects_sql_with_redis_backend() -> None:
    options = _community_options()
    options["db_backend"] = "redis"
    with pytest.raises(ValidationError, match="sql requires db_backend"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_rejects_community_without_async() -> None:
    options = _community_options()
    options["dispatch_mode"] = "sync"
    options["transport"] = "redis-sortedset"
    with pytest.raises(ValidationError, match="community supports"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_requires_community_images() -> None:
    options = _community_options()
    del options["images"]["gc"]
    with pytest.raises(ValidationError, match="images.gc is required"):
        parse_batch_gateway_options(options)
    options["images"]["gc"] = "img"
    options["images"]["extra"] = "img"
    with pytest.raises(ValidationError, match="unsupported fields: extra"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_rejects_images_on_operator() -> None:
    options = {
        "secret_name": "s",
        "file_storage": {"fs": {}},
        "images": dict(COMMUNITY_IMAGES),
    }
    with pytest.raises(ValidationError, match="images applies only"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_rejects_async_config_with_community() -> None:
    options = _community_options()
    options["async_config"] = {"redis": {"queuesConfig": []}}
    with pytest.raises(ValidationError, match="remove it from the profile"):
        parse_batch_gateway_options(options)


def test_parse_batch_gateway_options_rejects_bad_internal_route_auth() -> None:
    options = _community_options()
    options["internal_route_auth"] = "nope"
    with pytest.raises(ValidationError, match="internal_route_auth"):
        parse_batch_gateway_options(options)


def test_community_spec_uses_configured_objectives() -> None:
    plan = _plans("batch-gateway-async-sql")[0]
    spec = rhoai_batch_gateway_spec(plan)
    assert spec is not None
    assert spec.community_implementation
    assert spec.transport == "sql"
    assert spec.inference_objective == "batch-low"
    assert spec.interactive_objective == "interactive-high"
    assert spec.internal_route_auth is False
    profile_images = plan.deployment.options["batch_gateway"]["images"]
    assert spec.images == profile_images
    assert all(image.startswith("ghcr.io/llm-d/") and "@sha256:" in image
               for image in spec.images.values())
    assert spec.async_config == {}


def test_community_component_names_mirror_operator_convention() -> None:
    plan = _plans("batch-gateway-async-sql")[0]
    release = plan.deployment.release_name
    assert community_component_name(plan, "apiserver") == (
        f"{release}-batch-gateway-apiserver"
    )
    assert community_component_name(plan, "processor") == (
        f"{release}-batch-gateway-processor"
    )
    assert community_component_name(plan, "gc") == f"{release}-batch-gateway-gc"
    assert llm_d_async_name(plan) == f"{release}-llm-d-async"


def test_render_community_batch_gateway_manifests() -> None:
    plan = _plans("batch-gateway-async-sql")[0]
    spec = rhoai_batch_gateway_spec(plan)
    manifests = render_community_batch_gateway_manifests(
        plan, spec=spec, pool_name=TEST_POOL_NAME, model_url=TEST_MODEL_URL
    )
    by_filename = dict(manifests)
    assert [kind for _, kind_manifest in manifests for kind in [kind_manifest["kind"]]] == [
        "ServiceAccount",
        "ServiceAccount",
        "ServiceAccount",
        "ServiceAccount",
        "ConfigMap",
        "ConfigMap",
        "ConfigMap",
        "Role",
        "RoleBinding",
        "Deployment",
        "StatefulSet",
        "Deployment",
        "Deployment",
        "Service",
        "Service",
    ]
    for _, manifest in manifests:
        labels = manifest["metadata"]["labels"]
        assert labels["benchflow.io/batch-gateway"] == "true"
        assert labels["benchflow.io/release"] == plan.deployment.release_name
        assert manifest["metadata"]["namespace"] == plan.deployment.namespace

    release = plan.deployment.release_name
    processor_config = yaml.safe_load(
        by_filename["batch-processor-configmap.yaml"]["data"]["config.yaml"]
    )
    assert processor_config["dispatch_mode"] == "async"
    dispatch = processor_config["async_dispatch"]
    assert dispatch["transport"] == "sql"
    model_dispatch = dispatch["models"][plan.model.name]
    assert model_dispatch["inference_pool_name"] == TEST_POOL_NAME
    assert model_dispatch["inference_objective"] == "batch-low"
    assert model_dispatch["request_queue_name"] == f"llm-d-async:requests:{TEST_POOL_NAME}"
    assert model_dispatch["result_queue_name"] == f"llm-d-async:results:{TEST_POOL_NAME}"
    assert processor_config["db_client"] == {"type": "postgresql"}
    assert processor_config["file_client"]["type"] == "fs"

    statefulset = by_filename["batch-processor-statefulset.yaml"]
    assert statefulset["spec"]["serviceName"] == f"{release}-batch-gateway-processor"
    selector = statefulset["spec"]["selector"]["matchLabels"]
    assert selector["benchflow.io/release"] == release

    apiserver_service = by_filename["batch-apiserver-service.yaml"]
    assert apiserver_service["spec"]["ports"][0]["port"] == 8000

    gc_config = yaml.safe_load(
        by_filename["batch-gc-configmap.yaml"]["data"]["config.yaml"]
    )
    assert gc_config["reconciler"]["processor_statefulset"] == (
        f"{release}-batch-gateway-processor"
    )

    apiserver_config = yaml.safe_load(
        by_filename["batch-apiserver-configmap.yaml"]["data"]["config.yaml"]
    )
    assert apiserver_config["batch_api"]["pass_through_headers"] == ["Authorization"]

    role = by_filename["batch-gc-role.yaml"]
    assert role["rules"][0]["resources"] == ["pods"]


def test_render_community_llm_d_async_manifest_sql_url_from_secret() -> None:
    plan = _plans("batch-gateway-async-sql")[0]
    spec = rhoai_batch_gateway_spec(plan)
    manifest = render_community_llm_d_async_manifest(
        plan, spec=spec, pool_name=TEST_POOL_NAME, model_url=TEST_MODEL_URL
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]
    assert "--transport=sql" in args
    transport_config = json.loads(args[args.index("--transport-config") + 1])
    # The connection URL is injected via SQL_URL, never rendered inline.
    assert "url" not in transport_config
    assert transport_config["result_queue_name"] == (
        f"llm-d-async:results:{TEST_POOL_NAME}"
    )
    queue = transport_config["queues"][0]
    assert queue["queue_name"] == f"llm-d-async:requests:{TEST_POOL_NAME}"
    assert queue["igw_base_url"] == TEST_MODEL_URL
    assert queue["request_path_url"] == "/v1/chat/completions"
    assert queue["inference_objective"] == "batch-low"

    env = {entry["name"]: entry for entry in container["env"]}
    assert env["SQL_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "batch-gateway-sql-secrets",
        "key": "postgresql-url",
    }
    assert container["image"] == plan.deployment.options["batch_gateway"]["images"][
        "async"
    ]
    assert manifest["metadata"]["name"] == llm_d_async_name(plan)


def test_render_batch_inference_objective_manifest_interactive() -> None:
    plan = _plans("batch-gateway-async-sql")[0]
    spec = rhoai_batch_gateway_spec(plan)
    manifest = render_batch_inference_objective_manifest(
        plan,
        spec=spec,
        api_version="llm-d.ai/v1alpha2",
        pool_group="inference.networking.k8s.io",
        pool_name=TEST_POOL_NAME,
        name="interactive-high",
        priority=0,
        component="interactive-inference-objective",
    )
    assert manifest["metadata"]["name"] == "interactive-high"
    assert manifest["spec"]["priority"] == 0
    assert manifest["spec"]["poolRef"] == {
        "group": "inference.networking.k8s.io",
        "name": TEST_POOL_NAME,
    }
    # The batch objective keeps its sheddable default.
    batch = render_batch_inference_objective_manifest(
        plan,
        spec=spec,
        api_version="llm-d.ai/v1alpha2",
        pool_group="inference.networking.k8s.io",
        pool_name=TEST_POOL_NAME,
    )
    assert batch["metadata"]["name"] == "batch-low"
    assert batch["spec"]["priority"] == -1


def _deploy_mocks(monkeypatch, plan, *, applied: list, written: list):
    module = "benchflow.rhoai_batch"

    def fake_apply(kubectl_cmd, manifest, **kwargs):
        applied.append((manifest["kind"], manifest["metadata"]["name"]))
        return True

    def fake_write(manifests_dir, filename, manifest):
        written.append((filename, manifest["kind"]))

    monkeypatch.setattr(f"{module}._apply_manifest", fake_apply)
    monkeypatch.setattr(f"{module}._write_manifest", fake_write)
    monkeypatch.setattr(
        f"{module}._require_secret", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        f"{module}._ensure_namespace_gateway_label", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        f"{module}._ensure_file_storage_pvc", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        f"{module}.llmbatchgateway_api_version",
        lambda kubectl_cmd: TEST_API_VERSION,
    )
    monkeypatch.setattr(
        f"{module}.inference_objective_api_version",
        lambda kubectl_cmd: "llm-d.ai/v1alpha2",
    )
    monkeypatch.setattr(
        f"{module}.inference_pool_group",
        lambda kubectl_cmd: "inference.networking.k8s.io",
    )
    monkeypatch.setattr(
        f"{module}.load_rhoai_gateway_configuration",
        lambda kubectl_cmd: _gateway_config(),
    )
    monkeypatch.setattr(
        f"{module}.wait_for_rhoai_gateway_ready", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        f"{module}.discover_batch_internal_gateway_service",
        lambda *a, **kw: "internal-gw-svc",
    )
    monkeypatch.setattr(
        f"{module}.discover_inference_pool_name",
        lambda *a, **kw: TEST_POOL_NAME,
    )
    monkeypatch.setattr(
        f"{module}.wait_for_llmbatchgateway_ready", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        f"{module}.discover_batch_apiserver_service",
        lambda *a, **kw: "discovered-apiserver",
    )
    monkeypatch.setattr(
        f"{module}.wait_for_community_batch_gateway_ready", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        f"{module}.rhoai_batch_endpoint_url", lambda kubectl_cmd, plan: "https://batch"
    )


def test_deploy_community_skips_llmbatchgateway(monkeypatch, tmp_path) -> None:
    plan = _plans("batch-gateway-async-sql")[0]
    applied: list = []
    written: list = []
    _deploy_mocks(monkeypatch, plan, applied=applied, written=written)

    deploy_rhoai_batch_gateway(plan, kubectl_cmd="oc", manifests_dir=tmp_path)

    applied_kinds = [kind for kind, _ in applied]
    # Community path: no LLMBatchGateway CR is applied.
    assert "LLMBatchGateway" not in applied_kinds
    # llm-d-async renders with the conventional release-scoped name.
    deployment_names = [name for kind, name in applied if kind == "Deployment"]
    assert llm_d_async_name(plan) in deployment_names
    assert community_component_name(plan, "apiserver") in deployment_names
    assert community_component_name(plan, "gc") in deployment_names
    statefulsets = [name for kind, name in applied if kind == "StatefulSet"]
    assert statefulsets == [community_component_name(plan, "processor")]
    # The apiserver Service is conventional, not discovered.
    services = [name for kind, name in applied if kind == "Service"]
    assert community_component_name(plan, "apiserver") in services
    # Both objectives are applied: batch -1 and interactive 0.
    objectives = sorted(
        name for kind, name in applied if kind == "InferenceObjective"
    )
    assert objectives == ["batch-low", "interactive-high"]
    # internal_route_auth: false — the llm-route AuthPolicy is not applied
    # (the external batch-route AuthPolicy is a separate policy and stays).
    llm_auth = batch_llm_route_authpolicy_name(plan)
    assert llm_auth not in [name for kind, name in applied if kind == "AuthPolicy"]
    written_names = [f for f, kind in written if kind == "AuthPolicy"]
    assert "batch-llm-route-authpolicy.yaml" not in written_names


def test_deploy_operator_keeps_llmbatchgateway(monkeypatch, tmp_path) -> None:
    plan = _plans("batch-gateway-sync")[0]
    applied: list = []
    written: list = []
    _deploy_mocks(monkeypatch, plan, applied=applied, written=written)

    deploy_rhoai_batch_gateway(plan, kubectl_cmd="oc", manifests_dir=tmp_path)

    applied_kinds = [kind for kind, _ in applied]
    assert "LLMBatchGateway" in applied_kinds
    # Operator path: no community components are rendered.
    assert "StatefulSet" not in applied_kinds
    assert llm_d_async_name(plan) not in [
        name for kind, name in applied if kind == "Deployment"
    ]
    # internal_route_auth defaults to true — the llm-route AuthPolicy IS
    # applied alongside the batch-route one.
    authpolicies = sorted(name for kind, name in applied if kind == "AuthPolicy")
    assert batch_llm_route_authpolicy_name(plan) in authpolicies
