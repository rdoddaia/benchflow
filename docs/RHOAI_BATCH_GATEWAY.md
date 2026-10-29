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
- `profiles/deployment/rhoai/batch/batch-gateway-async-sql.yaml` — the
  community implementation over the PostgreSQL SQL queue transport (see
  "Community implementation and the SQL transport" below): BenchFlow renders
  the llm-d components directly, the EPP runs the SCR flow-control policy with
  in-flight eviction, and the runtime serves the YaRN-scaled Qwen3-32B copy at
  131072 context with report-era vLLM v0.26.0 flags.

Both pin `RHOAI-3.5.0` / channel `stable-3.5` — the stable line the upstream
deploy-rhoai.md recipe targets, whose operator already ships the batch-gateway
operator (`aigateway.batchGateway`) — and mirror the Diadochos serving runtime
(Qwen3-32B, TP8, vLLM 0.31.0, `max_num_seqs=256`, native 40960 context — the
legacy stack reaches 131072 only via a locally YaRN-scaled model copy). The
exact version pin keeps the installed-key fast path matching the cluster's
recorded platform state.

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
   The SQL transport needs only the `postgresql-url` key (a dedicated database
   on the shared PostgreSQL instance works — the transport creates its
   `async_*` tables on startup); the community Redis transport needs both keys.
   When `internal_route_auth: false` disables the llm-route AuthPolicy,
   clients reach the inference path without a Kubernetes token (legacy-stack
   driver parity; the external `/v1/batches` route keeps its own policy).
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

## Community implementation and the SQL transport

`options.batch_gateway.implementation: community` replaces the
LLMBatchGateway CR with BenchFlow-rendered community components
(`llm-d/llm-d-batch-gateway` apiserver/processor/gc plus the `llm-d-async`
dispatcher), mirroring the upstream `charts/batch-gateway` templates. It exists
because the RHOAI 3.5 CRD cannot express the SQL queue transport or in-flight
eviction. Constraints and behaviors:

- **Async dispatch only.** Sync dispatch stays on the operator path; the
  community path rejects `dispatch_mode: sync`.
- **Images are pinned by the profile.** `options.batch_gateway.images` maps
  `apiserver`, `processor`, `gc`, and `async` to digest-pinned references and
  is required. The SQL transport is main-branch only: its queue schema couples
  the processor (producer) and llm-d-async (consumer), so pin same-day
  digests.
- **`async_config` is derived.** The community renderer builds the whole
  processor `async_dispatch` block and the llm-d-async
  `--transport-config` from the release (model, pool, objective, queue names
  `llm-d-async:requests|results:<pool>`); a profile-supplied `async_config`
  is rejected.
- **Naming and cleanup are operator-identical.** Components are named
  `<release>-batch-gateway-{apiserver,processor,gc}` and
  `<release>-llm-d-async` and carry the same benchflow release labels, so the
  apiserver Service needs no discovery and `undeploy` deletes them with the
  same selector. The processor is a StatefulSet with a headless Service
  (chart shape); the gc watches pods through a small Role/RoleBinding.
- **Connection URLs never render into manifests.** The processor reads its
  database URL from the projected secret (`/etc/.secrets/postgresql-url`,
  the hardcoded upstream convention); llm-d-async gets `SQL_URL` (sql) or
  `REDIS_URL` (redis) via a `secretKeyRef`.
- **`transport: sql`** requires `implementation: community`,
  `dispatch_mode: async`, and `db_backend: postgresql` — the queues live in
  the batch gateway database and llm-d-async creates its `async_*` tables
  there on startup. Partitions (64 per queue) split evenly across processor
  replicas; no Redis/Valkey is involved.
- **Objectives.** The batch objective (`inference_objective`, priority -1) is
  rendered as usual; `interactive_objective` additionally renders a priority-0
  InferenceObjective so header-tagged interactive traffic keeps its protected
  band. The community processor stamps `x-gateway-inference-objective` on
  dispatched requests itself, and EPP v0.10.0 accepts both that name and the
  `x-llm-d-inference-objective` alias the legacy drivers send.
- **`internal_route_auth: false`** skips the llm-route AuthPolicy for
  driver parity with the legacy unauthenticated internal path.

The reference deployment is
`profiles/deployment/rhoai/batch/batch-gateway-async-sql.yaml` +
`experiments/rhoai/batch/batch-gateway-async-sql.yaml`, which pairs the
community SQL stack with the community EPP (v0.10.0, digest-pinned via
`spec.scheduler_image`) running the SCR policy (`enableEviction: true`, cap
128, TTL 45s — requires `epp_allow_experimental_plugins: true` because SCR
and eviction are Alpha) and the YaRN-scaled Qwen3-32B copy served at
`--max-model-len=131072` from a hostPath models PV pinned to one node
(`model_storage.model_subpath: Qwen-Qwen3-32B-yarn`; plain Qwen3-32B caps at
40960, so the env `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and the scaled copy are
both required).

## Option reference

`options.batch_gateway` (rhoai `distributed-default` profiles only):

| Field | Default | Notes |
| --- | --- | --- |
| `secret_name` | required | pre-existing Secret with `redis-url`, `postgresql-url` |
| `db_backend` | `postgresql` | `postgresql` or `redis` |
| `file_storage` | required | exactly one of `fs` (`claim_name`, `base_path`, `storage_class_name`) or `s3` (verbatim CR passthrough) |
| `dispatch_mode` | `sync` | `sync` or `async` |
| `implementation` | `operator` | `operator` (LLMBatchGateway CR) or `community` (rendered components; async only) |
| `transport` | `redis-sortedset` | `redis-sortedset` or `sql` (sql requires community + async + `db_backend: postgresql`) |
| `images` | — | community only; required; `apiserver`, `processor`, `gc`, `async` digest-pinned references |
| `interactive_objective` | unset | optional; renders an additional priority-0 InferenceObjective with this name |
| `internal_route_auth` | `true` | when false, skips the llm-route AuthPolicy (unauthenticated internal path) |
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
`prometheus-budget` gates) `gateParams.pool` and `gateParams.namespace`. The
community implementation derives the entire `async_config` instead — see
"Community implementation and the SQL transport".
