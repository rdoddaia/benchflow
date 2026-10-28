from __future__ import annotations


def run_batch_gateway_benchmark() -> None:
    """Not implemented: BenchFlow does not drive the /v1/batches API itself.

    The supported slice is deploying the model and the operator-managed batch
    gateway with a ``batch-gateway-*`` deployment profile. Drive the batch API
    with an external client that authenticates with a Kubernetes service-account
    token against the endpoint printed by the deploy stage. See
    docs/RHOAI_BATCH_GATEWAY.md for the full flow.
    """
    raise NotImplementedError(
        "BenchFlow does not benchmark the batch gateway /v1/batches API yet; "
        "point an external batch client at the endpoint printed by the deploy "
        "stage and authenticate with a Kubernetes service-account token"
    )
