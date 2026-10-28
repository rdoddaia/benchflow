# RHOAI Operator-Managed Batch Gateway

## Status

Implemented for BenchFlow RHOAI `LLMInferenceService` deployments
(`mode: distributed-default`): BenchFlow enables the operator, deploys the
per-release batch gateway stack, and prints the `/v1/batches` endpoint.
BenchFlow does **not** drive the `/v1/batches` API itself; point an external
batch client at the printed endpoint with a Kubernetes service-account token.
The placeholder for a first-party benchmark lives in
`src/benchflow/benchmark/batch_gateway.py`.

The implementation follows the upstream recipe
[deploy-rhoai.md](https://github.com/opendatahub-io/batch-gateway/blob/main/docs/guides/deploy-rhoai.md)
with one deliberate trim: the guide's `interactive-default` InferenceObjective
(priority 100) is not created, only the `batch-sheddable` objective
(priority -1) that the batch processor actually sends.

## Profiles

Two deployment profiles exercise the two dispatch modes:

- `profiles/deployment/rhoai/batch/batch-gateway-sync.yaml` — the processor
  sends inference requests directly to the release's internal Gateway over
  HTTP (`modelGateways.<model>.url`, plus `requestTimeout`, `maxRetries`,
  `initialBackoff`, `maxBackoff`).
- `profiles/deployment/rhoai/batch/batch-gateway-async.yaml` — the processor
  enqueues requests in Redis and the operator-deployed async-processor drains
  them (`asyncConfig`). The profile uses `local-max-concurrency` gates only, so
  it needs no Prometheus. Set the queue `gateType` to `prometheus-budget` and
  add `prometheusURL` for flow-control-based dispatch.

Both pin `RHOAI-3.5.0` / channel `stable-3.5` — the stable line the upstream
deploy-rhoai.md recipe targets, whose operator already ships the batch-gateway
operator (`aigateway.batchGateway`) — and mirror the Diadochos serving runtime
(Qwen3-32B, TP8, vLLM 0.31.0, `max_num_seqs=256`). The exact version pin keeps
the installed-key fast path matching the cluster's recorded platform state.

Example experiments: `experiments/rhoai/batch/batch-gateway-{sync,async}.yaml`.
They disable the benchmark stage because the batch drivers are external.

## Prerequisites

1. **RHOAI operator reachable at the pinned version.** Setup installs or
   reuses the `rhods-operator` subscription and then patches
   `spec.components.aigateway.{managementState,batchGateway.managementState}`
   to `Managed` on the `default-dsc` DataScienceCluster. The patch is a
   runtime `kubectl patch --type=merge`, not a change to BenchFlow's
   `datasciencecluster.yaml` asset, so a pre-existing cluster DSC is never
   reconciled to BenchFlow's template.
2. **The `batch-gateway-secrets` Secret** (name configurable via
   `options.batch_gateway.secret_name`) in the deployment namespace, with
   `redis-url` and `postgresql-url` keys. The backing Redis and PostgreSQL can
   live in any namespace the URLs reach. BenchFlow fails fast with a pointer to
   this document when the Secret is missing; it never creates it for you.
3. **Optional: Kuadrant.** When the Kuadrant APIs are absent, BenchFlow skips
   the batch AuthPolicies and the gateway runs without gateway-level
   authentication. The guide's TokenRateLimitPolicy is deliberately not
   applied: its 20 requests/minute demo limit would throttle benchmarks.

## What BenchFlow creates per release

All resources carry `benchflow.io/batch-gateway: "true"` plus
`benchflow.io/release` labels and are deleted by label selector at cleanup;
the release Gateway and the bootstrap Gateway are never touched.

- A ClusterIP-only **internal Gateway** in `openshift-ingress`, named
  `<release-gateway>-batch`, allowedRoutes restricted to namespaces labeled
  `llm-d.ai/gateway-route=true` (BenchFlow labels the deployment namespace).
- The **`batch-sheddable` InferenceObjective** (priority -1) referencing the
  InferencePool owned by the release's LLMInferenceService.
- The **`batch-llm-route` HTTPRoute** on the internal Gateway, matching
  `/<namespace>/<isvc>/v1/{completions,chat/completions}` and rewriting to the
  pool (async) or model service path.
- The **`LLMBatchGateway` CR** in the deployment namespace. The operator
  deploys the apiserver, processor, gc, and (in async mode) async-processor.
- The **`batch-route` HTTPRoute** on the release Gateway, exposing
  `/v1/batches` and `/v1/files` on the apiserver Service, plus a
  DestinationRule (TLS re-encrypt, `insecureSkipVerify`) only when
  `options.batch_gateway.tls.enabled` is set.
- A **20Gi ReadWriteMany PVC** for `file_storage.fs` when the claim does not
  exist. The default claim name is `<release>-batch-gateway-files`. Set
  `file_storage.fs.storage_class_name` when the cluster's default
  StorageClass cannot serve `ReadWriteMany` (Diadochos defaults to a block
  RWO class, so the profiles pin `rook-cephfs-fast`).

CRD group drift is handled by runtime discovery: `LLMBatchGateway`
(`batch.llm-d.ai` then `llm-d.ai`, preferring `v1alpha1`), `InferenceObjective`
(`inference.networking.x-k8s.io` then `llm-d.ai`, preferring `v1alpha2`), and
the InferencePool group (`k8s.io` then `x-k8s.io`).

## Platform setup and shared clusters

The `batchGateway` DSC component is deliberately **not** part of the setup key.
Batch and non-batch releases at the same RHOAI version share one installed
platform; enabling the component is additive and idempotent on both the setup
path and the installed-key fast path. BenchFlow records three state flags:

| Flag | Meaning |
| --- | --- |
| `batch_gateway_enabled` | BenchFlow patched the component to `Managed` |
| `batch_gateway_preexisting` | the component was already `Managed` |
| `datasciencecluster_managed` | BenchFlow created the DSC itself |

`setup rhoai` teardown patches the component back to `Removed` only when
BenchFlow enabled it on a DSC it did not create
(`batch_gateway_enabled` and neither `datasciencecluster_managed` nor
`batch_gateway_preexisting`). A component someone else enabled is never
disabled.

## Driving the batch API

The deploy stage ends by printing the batch endpoint
(`https://<release-gateway-listener-hostname>/v1/batches`, falling back to the
gateway status address when no listener hostname is set — the OpenShift router
routes by SNI, so the listener hostname is the client-usable address).
Authenticate with a Kubernetes
service-account token; when the Kuadrant AuthPolicies are enforced, the token's
user must also be allowed to `get` the target `llminferenceservices` resource
(the model-level SubjectAccessReview on the internal route).

The `/v1/batches` OpenAI-compatible surface (file upload, batch creation,
polling) is driven by your batch client, not by BenchFlow. When BenchFlow grows
a first-party batch benchmark it will live in
`src/benchflow/benchmark/batch_gateway.py`; until then that module raises
`NotImplementedError` on purpose.

## Option reference

`options.batch_gateway` (rhoai `distributed-default` profiles only):

| Field | Default | Notes |
| --- | --- | --- |
| `secret_name` | required | pre-existing Secret with `redis-url`, `postgresql-url` |
| `db_backend` | `postgresql` | `postgresql` or `redis` |
| `file_storage` | required | exactly one of `fs` (`claim_name`, `base_path`, `storage_class_name`) or `s3` (verbatim CR passthrough) |
| `dispatch_mode` | `sync` | `sync` or `async` |
| `api_server_replicas` | `1` | |
| `processor_replicas` | `1` | |
| `request_timeout` | `5m` | sync only |
| `max_retries` | `3` | sync only |
| `initial_backoff` | `1s` | sync only |
| `max_backoff` | `60s` | sync only |
| `inference_objective` | `<release>-batch-sheddable` | objective name |
| `async_config` | — | required for async, rejected for sync; CR-shaped passthrough |
| `gc_interval` | `30m` | |
| `tls` | unset | `{enabled, certManager}` passthrough; enables the DestinationRule |

BenchFlow injects release-derived `async_config` fields the profile cannot
know when they are absent: each queue's `name`, `igwBaseURL`, and (for
`prometheus-budget` gates) `gateParams.pool` and `gateParams.namespace`.
