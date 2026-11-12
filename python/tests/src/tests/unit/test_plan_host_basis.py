import copy
import dataclasses
import json
import logging

import pytest

from ambassador.compile import Compile
from ambassador.ir.irhttpmappinggroup import IRHTTPMappingGroup
from ambassador.plan_host_basis import (
    MAX_ARTIFACT_BYTES,
    MAX_CANDIDATES,
    BindingObligation,
    HostBasisSnapshot,
    HostCandidate,
    _SnapshotSecretHandler,
    plan_host_basis,
)
from ambassador.utils import SecretInfo

logger = logging.getLogger("ambassador")


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------


def _secret(name, namespace):
    return SecretInfo(
        name,
        namespace,
        "kubernetes.io/tls",
        tls_crt=f"fake-crt-{name}",
        tls_key=f"fake-key-{name}",
        decode_b64=False,
    )


# The baseline compile has no candidate Hosts, so the compiler resolves the
# fallback self-signed cert; the fixed snapshot must therefore contain it.
FALLBACK_KEY = "fallback-self-signed-cert.default"


def _secrets(**extra):
    secrets = {FALLBACK_KEY: _secret("fallback-self-signed-cert", "default")}
    secrets.update(extra)
    return secrets


def _listener(name, port, security_model, namespace_from, namespace="ambassador"):
    return f"""
---
apiVersion: getambassador.io/v3alpha1
kind: Listener
metadata:
  name: {name}
  namespace: {namespace}
spec:
  port: {port}
  protocol: HTTPS
  securityModel: {security_model}
  hostBinding:
    namespace:
      from: {namespace_from}
"""


def _mapping(name, namespace, hostname=None, labels=None, prefix="/"):
    labels_yaml = ""
    if labels:
        labels_yaml = "  labels:\n" + "\n".join(
            f"    {k}: {v}" for k, v in labels.items()
        )
    hostname_yaml = f"  hostname: {hostname}\n" if hostname else ""
    return f"""
---
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: {name}
  namespace: {namespace}
{labels_yaml}
spec:
{hostname_yaml}  prefix: {prefix}
  service: {name}:80
"""


def _host_manifest(
    name, namespace, hostname, *, tls_secret=None, mapping_selector=None
):
    tls_yaml = ""
    if tls_secret:
        tls_yaml = f"  tlsSecret:\n    name: {tls_secret}\n    namespace: {namespace}\n"
    selector_yaml = ""
    if mapping_selector:
        selector_yaml = "  mappingSelector:\n    matchLabels:\n" + "\n".join(
            f"      {k}: {v}" for k, v in mapping_selector.items()
        )
    return f"""
---
apiVersion: getambassador.io/v3alpha1
kind: Host
metadata:
  name: {name}
  namespace: {namespace}
spec:
  hostname: {hostname}
  requestPolicy:
    insecure:
      action: Route
{tls_yaml}{selector_yaml}"""


def _candidate(cid, manifest, cost, must_keep=False):
    return HostCandidate(
        candidate_id=cid,
        manifest=manifest,
        cost=cost,
        must_keep=must_keep,
    )


def _two_listener_base(*mappings):
    return (
        _listener("listener-8080", 8080, "XFP", "ALL")
        + _listener("listener-8443", 8443, "SECURE", "SELF")
        + "".join(mappings)
    )


def _obligation(oid, listener, group, forbid=False):
    return BindingObligation(
        obligation_id=oid,
        listener=listener,
        group=group,
        forbid=forbid,
    )


def _ids(artifact):
    return [host["candidate_id"] for host in artifact["selected_hosts"]]


def _witness_map(artifact):
    return {
        item["obligation_id"]: item["witnesses"] for item in artifact["obligations"]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanHostBasis:
    @pytest.mark.compilertest
    def test_cross_namespace_and_shared_listener_host(self):
        """SELF vs ALL namespace binding and one Host serving two Listeners."""
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "host-shop",
                    _host_manifest("h-shop", "shop", "shop.example.com"),
                    cost=5,
                ),
                _candidate(
                    "host-amb",
                    _host_manifest("h-amb", "ambassador", "shop.example.com"),
                    cost=3,
                ),
            ],
            obligations=[
                _obligation(
                    "all-listener-binding",
                    "ambassador/listener-8080",
                    "shop/GROUP: m1",
                ),
                _obligation(
                    "self-listener-binding",
                    "ambassador/listener-8443",
                    "shop/GROUP: m1",
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert _ids(artifact) == ["host-amb"]
        assert artifact["cost"]["total"] == 3
        assert artifact["cost"]["lower_bound"] == artifact["cost"]["upper_bound"] == 3
        assert artifact["bounds_proven_equal"] is True
        witnesses = _witness_map(artifact)
        # The ambassador-namespace Host binds both Listeners; the shop Host can
        # never match the SELF listener in namespace ambassador.
        assert witnesses["all-listener-binding"] == ["host-amb"]
        assert witnesses["self-listener-binding"] == ["host-amb"]
        assert artifact["counters"]["budget_exhausted"] is False

    @pytest.mark.compilertest
    def test_strict_selector_no_match_but_compat_switch_matches(self):
        """Strict selectors fail a partial label match; the legacy compatibility
        switch flips the same fixed snapshot to a match."""
        base = _two_listener_base(
            _mapping(
                "m-labeled",
                "shop",
                labels={"app": "web", "tier": "frontend"},
            )
        )
        candidate = _candidate(
            "selector-host",
            _host_manifest(
                "h-sel",
                "shop",
                "selector.example.com",
                mapping_selector={"app": "web", "env": "prod"},
            ),
            cost=2,
        )
        obligation = _obligation(
            "selector-binding",
            "ambassador/listener-8080",
            "shop/GROUP: m-labeled",
        )

        strict_snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[candidate],
            obligations=[obligation],
            secrets=_secrets(),
            files={},
        )

        strict_artifact = plan_host_basis(strict_snapshot, logger)
        # Under strict selectors the missing env=prod label means no candidate
        # can establish the binding; exhaustive search proves infeasibility.
        assert strict_artifact["status"] == "Infeasible"
        assert strict_artifact["strict_selectors_disabled"] is False
        assert strict_artifact["selected_hosts"] == []

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("DISABLE_STRICT_LABEL_SELECTORS", "true")
            compat_artifact = plan_host_basis(strict_snapshot, logger)

        assert compat_artifact["status"] == "Minimum"
        assert compat_artifact["strict_selectors_disabled"] is True
        assert _ids(compat_artifact) == ["selector-host"]
        # The switch is part of the semantic input, so the digest must change.
        assert strict_artifact["input_digest"] != compat_artifact["input_digest"]

    @pytest.mark.compilertest
    def test_forbidden_binding_excludes_low_cost_candidate(self):
        """A cheap candidate that establishes a forbidden binding is excluded,
        forcing a more expensive subset."""
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "cheap-forbidden",
                    _host_manifest("h-cheap", "ambassador", "shop.example.com"),
                    cost=1,
                ),
                _candidate(
                    "pricy-allowed",
                    _host_manifest("h-ok", "shop", "shop.example.com"),
                    cost=9,
                ),
            ],
            obligations=[
                _obligation(
                    "required",
                    "ambassador/listener-8080",
                    "shop/GROUP: m1",
                ),
                _obligation(
                    "forbidden",
                    "ambassador/listener-8443",
                    "shop/GROUP: m1",
                    forbid=True,
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert _ids(artifact) == ["pricy-allowed"]
        assert artifact["cost"]["total"] == 9
        witnesses = _witness_map(artifact)
        assert witnesses["required"] == ["pricy-allowed"]
        assert witnesses["forbidden"] == []

    @pytest.mark.compilertest
    def test_tls_filter_chain_combination_conflict_is_infeasible(self):
        """Each Host alone compiles, but the forced combination collides on a
        filter chain (same SNI, different TLSContext) and is infeasible."""
        base = _two_listener_base(_mapping("m-tls", "shop", hostname="tls.example.com"))
        secrets = _secrets(
            **{
                "secret-a.shop": _secret("secret-a", "shop"),
                "secret-b.shop": _secret("secret-b", "shop"),
            }
        )
        host_a = _candidate(
            "host-a",
            _host_manifest(
                "h-a",
                "shop",
                "tls.example.com",
                tls_secret="secret-a",
            ),
            cost=1,
            must_keep=True,
        )
        host_b = _candidate(
            "host-b",
            _host_manifest(
                "h-b",
                "shop",
                "tls.example.com",
                tls_secret="secret-b",
            ),
            cost=1,
            must_keep=True,
        )
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[host_a, host_b],
            obligations=[
                _obligation(
                    "tls-binding",
                    "ambassador/listener-8080",
                    "shop/GROUP: m-tls",
                ),
            ],
            secrets=secrets,
            files={},
        )

        # Single-binding feasibility: each Host compiles cleanly on its own.
        for host in (host_a, host_b):
            single = HostBasisSnapshot(
                base_manifests=base,
                candidates=[host],
                obligations=[
                    _obligation(
                        "tls-binding",
                        "ambassador/listener-8080",
                        "shop/GROUP: m-tls",
                    ),
                ],
                secrets=secrets,
                files={},
            )
            single_artifact = plan_host_basis(single, logger)
            assert single_artifact["status"] == "Minimum"

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Infeasible"
        assert artifact["selected_hosts"] == []
        # The failure is recorded against the exact full selection only.
        assert artifact["oracle_failures"]
        failed_ids = set()
        for failure in artifact["oracle_failures"]:
            failed_ids.update(failure["selected_hosts"])
        assert failed_ids == {"host-a", "host-b"}
        assert artifact["counters"]["evaluations"] >= 1

    @pytest.mark.compilertest
    def test_oracle_failure_excludes_only_the_verified_selection(self):
        """The cheapest covering selection fails the real Compile (TLS context
        collision on one SNI); the planner must recover with a compiling
        alternative instead of extrapolating the failure to other selections."""
        base = _two_listener_base(_mapping("m1", "shop", hostname="shared.example.com"))
        secrets = _secrets(
            **{
                "secret-a.ambassador": _secret("secret-a", "ambassador"),
                "secret-b.other": _secret("secret-b", "other"),
            }
        )
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "partner",
                    _host_manifest(
                        "h-partner",
                        "other",
                        "shared.example.com",
                        tls_secret="secret-b",
                    ),
                    cost=0,
                    must_keep=True,
                ),
                _candidate(
                    "cheap-tls",
                    _host_manifest(
                        "h-cheap",
                        "ambassador",
                        "shared.example.com",
                        tls_secret="secret-a",
                    ),
                    cost=1,
                ),
                _candidate(
                    "safe-cleartext",
                    _host_manifest(
                        "h-safe",
                        "ambassador",
                        "shared.example.com",
                    ),
                    cost=9,
                ),
            ],
            obligations=[
                _obligation(
                    "o1",
                    "ambassador/listener-8080",
                    "shop/GROUP: m1",
                ),
                _obligation(
                    "o2",
                    "ambassador/listener-8443",
                    "shop/GROUP: m1",
                ),
            ],
            secrets=secrets,
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert set(_ids(artifact)) == {"partner", "safe-cleartext"}
        assert artifact["cost"]["total"] == 9
        failed = [
            set(failure["selected_hosts"]) for failure in artifact["oracle_failures"]
        ]
        # Every rejected selection is excluded on its own verified failure and
        # each genuinely contains the colliding TLS pair; the failure is never
        # extrapolated onto selections (like the winner) that lack that pair.
        assert failed
        assert all({"partner", "cheap-tls"} <= hosts for hosts in failed)
        assert all(
            "safe-cleartext" not in hosts or "cheap-tls" in hosts for hosts in failed
        )
        assert set(_ids(artifact)) not in failed

    @pytest.mark.compilertest
    def test_zero_cost_tie_breaks_on_host_count_then_identity(self):
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "zb",
                    _host_manifest("zb", "shop", "shop.example.com"),
                    cost=0,
                ),
                _candidate(
                    "za",
                    _host_manifest("za", "shop", "shop.example.com"),
                    cost=0,
                ),
                _candidate(
                    "zc-unrelated",
                    _host_manifest("zc", "other", "elsewhere.example.com"),
                    cost=0,
                ),
            ],
            obligations=[
                _obligation(
                    "o1",
                    "ambassador/listener-8080",
                    "shop/GROUP: m1",
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert artifact["cost"]["total"] == 0
        assert artifact["host_count"] == 1
        assert _ids(artifact) == ["za"]
        assert artifact["bounds_proven_equal"] is True

    @pytest.mark.compilertest
    def test_listener_hostbinding_matchlabels_is_reused(self):
        """A hostBinding.selector.matchLabels gate is enforced through the real
        IRListener.matches_host predicate."""
        base = """
---
apiVersion: getambassador.io/v3alpha1
kind: Listener
metadata:
  name: labeled-listener
  namespace: ambassador
spec:
  port: 8081
  protocol: HTTPS
  securityModel: XFP
  hostBinding:
    namespace:
      from: ALL
    selector:
      matchLabels:
        app: edge
""" + _mapping("m-edge", "shop", hostname="edge.example.com")
        edge_host = """
---
apiVersion: getambassador.io/v3alpha1
kind: Host
metadata:
  name: h-edge
  namespace: shop
  labels:
    app: edge
spec:
  hostname: edge.example.com
  requestPolicy:
    insecure:
      action: Route
"""
        plain_host = _host_manifest("h-plain", "shop", "edge.example.com")
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate("edge", edge_host, cost=3),
                _candidate("plain", plain_host, cost=1),
            ],
            obligations=[
                _obligation(
                    "labeled-binding",
                    "ambassador/labeled-listener",
                    "shop/GROUP: m-edge",
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert _ids(artifact) == ["edge"]
        assert _witness_map(artifact)["labeled-binding"] == ["edge"]

    @pytest.mark.compilertest
    def test_mapping_selector_and_hostname_must_both_match(self):
        """Under strict selectors, when both a mappingSelector and a Mapping
        hostname are present, both must match (no partial credit)."""
        base = _two_listener_base(
            _mapping(
                "m-combo",
                "shop",
                hostname="combo.example.com",
                labels={"app": "web"},
            )
        )
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "both",
                    _host_manifest(
                        "h-both",
                        "shop",
                        "combo.example.com",
                        mapping_selector={"app": "web"},
                    ),
                    cost=4,
                ),
                _candidate(
                    "selector-only",
                    _host_manifest(
                        "h-wrong-host",
                        "shop",
                        "wrong.example.com",
                        mapping_selector={"app": "web"},
                    ),
                    cost=1,
                ),
            ],
            obligations=[
                _obligation(
                    "combo-binding",
                    "ambassador/listener-8080",
                    "shop/GROUP: m-combo",
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert _ids(artifact) == ["both"]
        assert _witness_map(artifact)["combo-binding"] == ["both"]

    @pytest.mark.compilertest
    def test_synthetic_mapping_group_semantics_are_reused(self):
        """Obligations against a synthetic probe group reuse the real
        IRHost.matches_httpgroup behavior, which always matches those groups."""
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "real-host",
                    _host_manifest("h1", "shop", "shop.example.com"),
                    cost=5,
                ),
            ],
            obligations=[
                _obligation(
                    "synthetic-binding",
                    "ambassador/listener-8080",
                    "default/GROUP: internal_readiness_probe_mapping",
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        assert _ids(artifact) == ["real-host"]
        assert _witness_map(artifact)["synthetic-binding"] == ["real-host"]

    @pytest.mark.compilertest
    def test_missing_snapshot_secret_is_incomplete_not_infeasible(self):
        base = _two_listener_base(_mapping("m-tls", "shop", hostname="tls.example.com"))
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "tls-host",
                    _host_manifest(
                        "h-tls",
                        "shop",
                        "tls.example.com",
                        tls_secret="missing-secret",
                    ),
                    cost=1,
                ),
            ],
            obligations=[
                _obligation(
                    "tls-binding",
                    "ambassador/listener-8080",
                    "shop/GROUP: m-tls",
                ),
            ],
            # Fallback present, but the Host's own secret is absent.
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Incomplete"
        assert "missing-secret.shop" in artifact["incomplete"]["missing_secrets"]
        assert artifact["selected_hosts"] == []
        assert artifact["cost"]["lower_bound"] is None

    @pytest.mark.compilertest
    def test_budget_exhaustion_keeps_best_known_without_minimum_claim(self):
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "h2",
                    _host_manifest("h2", "other", "other.example.com"),
                    cost=1,
                ),
                _candidate(
                    "h3",
                    _host_manifest("h3", "ambassador", "shop.example.com"),
                    cost=3,
                ),
                _candidate(
                    "h1",
                    _host_manifest("h1", "shop", "shop.example.com"),
                    cost=5,
                ),
            ],
            obligations=[
                _obligation(
                    "o1",
                    "ambassador/listener-8080",
                    "shop/GROUP: m1",
                ),
                _obligation(
                    "o2",
                    "ambassador/listener-8443",
                    "shop/GROUP: m1",
                ),
            ],
            secrets=_secrets(),
            files={},
            max_evaluations=1,
        )

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "BestKnown"
        assert artifact["counters"]["budget_exhausted"] is True
        assert artifact["bounds_proven_equal"] is False
        assert artifact["cost"]["upper_bound"] is not None
        assert artifact["cost"]["lower_bound"] is not None
        assert artifact["cost"]["lower_bound"] < artifact["cost"]["upper_bound"]

    @pytest.mark.compilertest
    def test_brute_force_enumeration_matches_planner_optimum(self):
        """Independently enumerate every subset with the real Compile oracle
        and confirm the planner returns the brute-force optimum."""
        base = _two_listener_base(
            _mapping("m1", "shop", hostname="shop.example.com"),
            _mapping("m2", "shop", hostname="api.example.com"),
        )
        candidates = [
            _candidate(
                "c1",
                _host_manifest("h1", "shop", "shop.example.com"),
                cost=4,
            ),
            _candidate(
                "c2",
                _host_manifest("h2", "shop", "api.example.com"),
                cost=6,
            ),
            _candidate(
                "c3",
                _host_manifest("h3", "ambassador", "shop.example.com"),
                cost=2,
            ),
        ]
        obligations = [
            _obligation(
                "o1",
                "ambassador/listener-8080",
                "shop/GROUP: m1",
            ),
            _obligation(
                "o2",
                "ambassador/listener-8080",
                "shop/GROUP: m2",
            ),
            _obligation(
                "o3",
                "ambassador/listener-8443",
                "shop/GROUP: m1",
            ),
        ]
        secrets = _secrets()
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=candidates,
            obligations=obligations,
            secrets=secrets,
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)
        assert artifact["status"] == "Minimum"

        # Independent brute force: real all-candidate Compile for the matrix,
        # then a real isolated Compile for every structurally covering subset.
        all_yaml = base + "".join(c.manifest for c in candidates)
        analysis = Compile(
            logger,
            all_yaml,
            cache=None,
            secret_handler=_SnapshotSecretHandler(logger, secrets),
            k8s=True,
        )
        analysis_ir = analysis["ir"]
        assert analysis_ir.aconf.errors == {}

        listeners = {
            f"{item.namespace}/{item.name}": item
            for item in analysis_ir.listeners.values()
        }
        groups = {
            f"{item.namespace}/{item.name}": item
            for item in analysis_ir.groups.values()
            if isinstance(item, IRHTTPMappingGroup)
        }
        hosts = [analysis_ir.hosts[doc_name] for doc_name in ("h1", "h2", "h3")]

        def covers(mask):
            for obligation in obligations:
                listener = listeners[obligation.listener]
                group = groups[obligation.group]
                if not any(
                    listener.matches_host(hosts[idx])
                    and hosts[idx].matches_httpgroup(group)
                    for idx in range(3)
                    if mask & (1 << idx)
                ):
                    return False
            return True

        brute_optimum = None
        for mask in range(1, 1 << len(candidates)):
            if not covers(mask):
                continue
            yaml_text = base + "".join(
                candidates[idx].manifest
                for idx in range(len(candidates))
                if mask & (1 << idx)
            )
            result = Compile(
                logger,
                yaml_text,
                cache=None,
                secret_handler=_SnapshotSecretHandler(logger, secrets),
                k8s=True,
            )
            if result["ir"].aconf.errors:
                continue
            cost = sum(
                candidates[idx].cost
                for idx in range(len(candidates))
                if mask & (1 << idx)
            )
            key = (
                cost,
                bin(mask).count("1"),
                tuple(sorted(_mask_refs(mask, candidates))),
            )
            if brute_optimum is None or key < brute_optimum[0]:
                brute_optimum = (key, mask)

        assert brute_optimum is not None
        planned_mask = sum(
            1 << idx
            for idx, candidate in enumerate(candidates)
            if candidate.candidate_id in _ids(artifact)
        )
        assert planned_mask == brute_optimum[1]
        assert artifact["cost"]["total"] == brute_optimum[0][0]

    @pytest.mark.compilertest
    def test_brute_force_agrees_on_proven_infeasibility(self):
        base = _two_listener_base(
            _mapping("m1", "shop", hostname="shop.example.com"),
            _mapping("m2", "shop", hostname="unreachable.example.com"),
        )
        candidates = [
            _candidate(
                "c1",
                _host_manifest("h1", "shop", "shop.example.com"),
                cost=1,
            ),
        ]
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=candidates,
            obligations=[
                _obligation(
                    "uncoverable",
                    "ambassador/listener-8080",
                    "shop/GROUP: m2",
                ),
            ],
            secrets=_secrets(),
            files={},
        )

        artifact = plan_host_basis(snapshot, logger)
        assert artifact["status"] == "Infeasible"
        assert artifact["counters"]["budget_exhausted"] is False

        # The empty selection is a valid compile input and defines the baseline
        # error set (an unmatched Listener reports its own baseline error).
        baseline = Compile(
            logger,
            base,
            cache=None,
            secret_handler=_SnapshotSecretHandler(logger, _secrets()),
            k8s=True,
        )
        baseline_errors = baseline["ir"].aconf.errors

        # Brute force over the finite candidate space reaches the same verdict:
        # every selection compiles, but none establishes the required binding.
        for mask in range(0, 1 << len(candidates)):
            yaml_text = base + "".join(
                candidates[idx].manifest
                for idx in range(len(candidates))
                if mask & (1 << idx)
            )
            result = Compile(
                logger,
                yaml_text,
                cache=None,
                secret_handler=_SnapshotSecretHandler(logger, _secrets()),
                k8s=True,
            )
            assert result["ir"].aconf.errors == baseline_errors

    @pytest.mark.compilertest
    def test_inputs_are_not_mutated_and_artifact_is_bounded(self):
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))
        secrets = _secrets()
        files = {}
        snapshot = HostBasisSnapshot(
            base_manifests=base,
            candidates=[
                _candidate(
                    "c1",
                    _host_manifest("h1", "shop", "shop.example.com"),
                    cost=4,
                    must_keep=True,
                ),
            ],
            obligations=[
                _obligation(
                    "o1",
                    "ambassador/listener-8080",
                    "shop/GROUP: m1",
                ),
            ],
            secrets=secrets,
            files=files,
        )
        snapshot_copy = copy.deepcopy(snapshot)
        secrets_before = {key: info.to_dict() for key, info in secrets.items()}

        artifact = plan_host_basis(snapshot, logger)

        assert artifact["status"] == "Minimum"
        # SecretInfo has no value equality, so compare primitive fields and
        # serialized secret contents rather than object identity.
        assert snapshot.base_manifests == snapshot_copy.base_manifests
        assert snapshot.candidates == snapshot_copy.candidates
        assert snapshot.obligations == snapshot_copy.obligations
        assert snapshot.files == snapshot_copy.files
        assert snapshot.max_evaluations == snapshot_copy.max_evaluations
        assert {
            key: info.to_dict() for key, info in snapshot.secrets.items()
        } == secrets_before
        assert snapshot.files == files
        assert files == {}
        assert snapshot.candidates[0].manifest == snapshot_copy.candidates[0].manifest
        serialized = json.dumps(artifact, default=str)
        assert len(serialized.encode("utf-8")) <= MAX_ARTIFACT_BYTES

    @pytest.mark.compilertest
    def test_digest_stable_for_identical_inputs(self):
        base = _two_listener_base(_mapping("m1", "shop", hostname="shop.example.com"))

        def build():
            return HostBasisSnapshot(
                base_manifests=base,
                candidates=[
                    _candidate(
                        "c1",
                        _host_manifest("h1", "shop", "shop.example.com"),
                        cost=4,
                    ),
                ],
                obligations=[
                    _obligation(
                        "o1",
                        "ambassador/listener-8080",
                        "shop/GROUP: m1",
                    ),
                ],
                secrets=_secrets(),
                files={},
            )

        first = plan_host_basis(build(), logger)
        second = plan_host_basis(build(), logger)
        assert first["input_digest"] == second["input_digest"]

        changed = build()
        # Frozen dataclasses require replacement rather than mutation.
        changed = dataclasses.replace(
            changed,
            candidates=[
                dataclasses.replace(changed.candidates[0], cost=5),
            ],
        )
        third = plan_host_basis(changed, logger)
        assert third["input_digest"] != first["input_digest"]

    @pytest.mark.compilertest
    def test_invalid_inputs_are_rejected(self):
        base = _two_listener_base()
        good_host = _host_manifest("h1", "shop", "shop.example.com")
        obligation = _obligation("o1", "ambassador/listener-8080", "shop/GROUP: m1")

        with pytest.raises(ValueError):
            plan_host_basis(
                HostBasisSnapshot(
                    base_manifests=base,
                    candidates=[_candidate("c1", good_host, cost=-1)],
                    obligations=[obligation],
                    secrets=_secrets(),
                    files={},
                ),
                logger,
            )

        with pytest.raises(ValueError):
            plan_host_basis(
                HostBasisSnapshot(
                    base_manifests=base,
                    candidates=[
                        _candidate("c1", good_host, 1),
                        _candidate("c1", good_host, 2),
                    ],
                    obligations=[obligation],
                    secrets=_secrets(),
                    files={},
                ),
                logger,
            )

        with pytest.raises(ValueError):
            plan_host_basis(
                HostBasisSnapshot(
                    base_manifests=base,
                    candidates=[_candidate("c1", good_host, 1)],
                    obligations=[
                        _obligation("o1", "missing-listener", "shop/GROUP: m1"),
                    ],
                    secrets=_secrets(),
                    files={},
                ),
                logger,
            )

        with pytest.raises(ValueError):
            plan_host_basis(
                HostBasisSnapshot(
                    base_manifests=base,
                    candidates=[
                        _candidate(
                            "c1",
                            good_host.replace("kind: Host", "kind: Mapping"),
                            1,
                        ),
                    ],
                    obligations=[obligation],
                    secrets=_secrets(),
                    files={},
                ),
                logger,
            )

        with pytest.raises(ValueError):
            plan_host_basis(
                HostBasisSnapshot(
                    base_manifests=base,
                    candidates=[_candidate("c1", good_host, 1)],
                    obligations=[obligation],
                    secrets=_secrets(),
                    files={},
                    max_evaluations=0,
                ),
                logger,
            )

    @pytest.mark.compilertest
    def test_candidate_count_is_hard_limited(self):
        base = _two_listener_base()
        obligation = _obligation("o1", "ambassador/listener-8080", "shop/GROUP: m1")
        too_many = [
            _candidate(
                f"c{i}",
                _host_manifest(f"h{i}", "shop", f"host{i}.example.com"),
                1,
            )
            for i in range(MAX_CANDIDATES + 1)
        ]
        with pytest.raises(ValueError):
            plan_host_basis(
                HostBasisSnapshot(
                    base_manifests=base,
                    candidates=too_many,
                    obligations=[obligation],
                    secrets=_secrets(),
                    files={},
                ),
                logger,
            )


def _mask_refs(mask, candidates):
    refs = []
    for idx, candidate in enumerate(candidates):
        if mask & (1 << idx):
            refs.append(candidate.candidate_id)
    return refs
