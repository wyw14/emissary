import copy
import json
import logging

import pytest

from ambassador.compile import Compile
from ambassador.compile_closure import (
    BUNDLE_FORMAT,
    ClosureError,
    ClosureFormatError,
    ClosureLimitError,
    ClosureRoot,
    ManifestSecretHandler,
    export_compile_closure,
    import_compile_closure,
    manifest_file_checker,
    validate_relative_safety,
)

logger = logging.getLogger("ambassador")


#############################################################################
# Fixtures
#############################################################################


SECRET_MATERIAL = {
    "tls-crt.other": {"tls_crt": "upstream-cert-pem", "tls_key": "upstream-key-pem"},
    "host-cert.secure-ns": {"tls_crt": "host-cert-pem", "tls_key": "host-key-pem"},
}


def _watt_snapshot() -> dict:
    return {
        "Kubernetes": {
            "Listener": [
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Listener",
                    "metadata": {"name": "l8080", "namespace": "default"},
                    "spec": {
                        "port": 8080,
                        "protocol": "HTTPS",
                        "securityModel": "XFP",
                        "hostBinding": {"namespace": {"from": "ALL"}},
                    },
                }
            ],
            "Module": [
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Module",
                    "metadata": {"name": "ambassador", "namespace": "default"},
                    "spec": {"config": {}},
                }
            ],
            # A user-named endpoint resolver, plus the aliased built-in name.
            "KubernetesEndpointResolver": [
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "KubernetesEndpointResolver",
                    "metadata": {"name": "myendpoint", "namespace": "default"},
                    "spec": {},
                },
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "KubernetesEndpointResolver",
                    "metadata": {"name": "endpoint", "namespace": "default"},
                    "spec": {},
                },
            ],
            "TLSContext": [
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "TLSContext",
                    "metadata": {"name": "upstream-tls", "namespace": "default"},
                    # Cross-namespace secret reference via the name.namespace form.
                    "spec": {"hosts": ["example.com"], "secret": "tls-crt.other"},
                }
            ],
            "Host": [
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Host",
                    "metadata": {"name": "api.example.com", "namespace": "default"},
                    "spec": {
                        "hostname": "api.example.com",
                        "tlsSecret": {"name": "host-cert", "namespace": "secure-ns"},
                    },
                }
            ],
            "Mapping": [
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Mapping",
                    "metadata": {"name": "m1", "namespace": "default"},
                    "spec": {
                        "hostname": "*",
                        "prefix": "/a/",
                        "service": "qotm",
                        "resolver": "myendpoint",
                    },
                },
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Mapping",
                    "metadata": {"name": "m2", "namespace": "default"},
                    "spec": {
                        "hostname": "*",
                        "prefix": "/b/",
                        "service": "qotm",
                        "resolver": "myendpoint",
                    },
                },
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Mapping",
                    "metadata": {"name": "m3", "namespace": "other"},
                    "spec": {
                        "hostname": "*",
                        "prefix": "/c/",
                        "service": "https://secure:443",
                        "tls": "upstream-tls",
                    },
                },
                {
                    "apiVersion": "getambassador.io/v3alpha1",
                    "kind": "Mapping",
                    "metadata": {"name": "unused-mapping", "namespace": "default"},
                    "spec": {
                        "hostname": "*",
                        "prefix": "/z/",
                        "service": "unused-svc",
                    },
                },
            ],
            "service": [
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": "qotm",
                        "namespace": "default",
                        "labels": {"app.kubernetes.io/component": "ambassador-service"},
                    },
                    "spec": {
                        "ports": [
                            {
                                "name": "http",
                                "port": 80,
                                "targetPort": 8080,
                                "protocol": "TCP",
                            }
                        ],
                        "selector": {"app": "qotm"},
                    },
                }
            ],
            "Endpoints": [
                {
                    "apiVersion": "v1",
                    "kind": "Endpoints",
                    "metadata": {"name": "qotm", "namespace": "default"},
                    "subsets": [
                        {
                            "addresses": [
                                {"ip": "10.0.0.1"},
                                {"ip": "10.0.0.2"},
                            ],
                            "ports": [
                                {"name": "http", "port": 8080, "protocol": "TCP"}
                            ],
                        }
                    ],
                }
            ],
        }
    }


def _compile(watt: dict, material=None, file_checker=None):
    material = SECRET_MATERIAL if material is None else material
    handler = ManifestSecretHandler(logger, material)
    checker = file_checker or (lambda path: True)
    result = Compile(
        logger,
        json.dumps(watt),
        k8s=False,
        secret_handler=handler,
        file_checker=checker,
    )
    return result, handler


def _global_errors(ir) -> dict:
    errors = dict(ir.aconf.errors)
    errors.pop("-global-", None)  # container-environment notice, unrelated
    return errors


def _resource_identities(bundle):
    return {(r["kind"], r["name"], r["namespace"]) for r in bundle["resources"]}


def _replay(bundle, material=SECRET_MATERIAL, **kwargs):
    handler = ManifestSecretHandler(logger, material)
    return import_compile_closure(bundle, secret_handler=handler, **kwargs)


#############################################################################
# Tests
#############################################################################


@pytest.mark.compilertest
def test_two_roots_share_one_cluster_and_no_unused_leak():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]
    assert not _global_errors(ir)

    roots = [
        ClosureRoot("Mapping", "m1", "default"),
        ClosureRoot("Mapping", "m2", "default"),
    ]
    bundle = export_compile_closure(ir, roots)

    assert bundle["completeness"] == "Complete"
    assert bundle["incomplete_reasons"] == []

    # Both roots resolve to a single shared cluster: identity is preserved.
    clusters = bundle["signature"]["clusters"]
    assert len(clusters) == 1
    shared_cluster = next(iter(clusters))
    assert shared_cluster == "cluster_qotm_myendpoint_default"

    group_clusters = {
        mapping["cluster"]
        for group in bundle["signature"]["groups"]
        for mapping in group["mappings"]
    }
    assert group_clusters == {shared_cluster}

    # Real endpoint targets parsed by the IR are carried in the signature.
    cluster = clusters[shared_cluster]
    assert {(t["ip"], t["port"]) for t in cluster["targets"]} == {
        ("10.0.0.1", 8080),
        ("10.0.0.2", 8080),
    }

    # Only closure resources are bundled; the unrelated mapping is excluded.
    identities = _resource_identities(bundle)
    assert ("Mapping", "m1", "default") in identities
    assert ("Mapping", "m2", "default") in identities
    assert ("Mapping", "unused-mapping", "default") not in identities
    assert ("Mapping", "m3", "other") not in identities
    assert not any(kind == "Listener" for kind, _, _ in identities)

    # Unused resources are reported explicitly, as identities only.
    unused = bundle["references"]["unused"]
    assert {
        "kind": "Mapping",
        "name": "unused-mapping",
        "namespace": "default",
    } in unused

    # Replay through the real Compile, then field-by-field comparison.
    replay = _replay(bundle)
    assert replay.complete, replay.reasons
    assert replay.diff == []
    assert "xds" in replay.result

    # No unrelated user resource leaks into the replayed IR.  The compiler's
    # own readiness/liveness/diagnostics probe mappings are synthesized
    # internally by every compile (including the original) and are not user
    # resources from the snapshot.
    replay_names = {
        mapping.name
        for group in replay.result["ir"].groups.values()
        for mapping in group.mappings
        if not mapping.name.startswith("internal_")
    }
    assert replay_names == {"m1", "m2"}
    assert "unused-mapping" not in replay_names
    assert "m3" not in replay_names


@pytest.mark.compilertest
def test_cross_namespace_secret_is_handle_and_digest_only():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    bundle = export_compile_closure(
        ir, [ClosureRoot("Host", "api.example.com", "default")]
    )

    secrets = {
        entry["name"] + "." + entry["namespace"]: entry
        for entry in bundle["external"]["secrets"]
    }
    host_secret = secrets["host-cert.secure-ns"]

    # Real cross-namespace identity is preserved on the external handle.
    assert host_secret["handle"] == "k8s:v1/Secret/secure-ns/host-cert"
    assert host_secret["status"] == "Present"
    assert host_secret["digests"]["tls_crt"].startswith("sha256:")
    assert host_secret["digests"]["tls_key"].startswith("sha256:")

    # No credential material anywhere in the bundle.
    raw = json.dumps(bundle)
    for material in ("host-cert-pem", "host-key-pem", "upstream-cert-pem"):
        assert material not in raw

    # Secrets never appear as bundle resources.
    assert not any(entry["kind"] == "Secret" for entry in bundle["resources"])

    # The implicit host context is part of the closure.
    assert "api.example.com-context" in bundle["signature"]["tls_contexts"]

    # Matching handler: exact replay.
    replay = _replay(bundle)
    assert replay.complete, replay.reasons
    assert replay.diff == []

    # Handler with wrong content (digest mismatch) is honestly rejected.
    bad_material = copy.deepcopy(SECRET_MATERIAL)
    bad_material["host-cert.secure-ns"]["tls_crt"] = "different-cert"
    replay_bad = _replay(bundle, material=bad_material)
    assert not replay_bad.complete
    assert any("digest does not match" in reason for reason in replay_bad.reasons)

    # No handler at all: Incomplete, and no fake-secret success claim.
    replay_missing = import_compile_closure(bundle)
    assert replay_missing.status == "Incomplete"
    assert any(
        "was not provided by the SecretHandler" in reason
        for reason in replay_missing.reasons
    )


@pytest.mark.compilertest
def test_resolver_alias_keeps_real_identity():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m1", "default")])

    resolvers = bundle["signature"]["resolvers"]
    # The cluster uses the alias name "myendpoint"; that name must survive
    # rather than being normalized to a built-in resolver name.
    assert "myendpoint" in resolvers
    assert resolvers["myendpoint"]["kind"] == "KubernetesEndpointResolver"

    # Only the user-supplied resolver resource travels in the bundle;
    # synthesized aliases/defaults are recreated by the compiler on replay.
    resolver_resources = [
        entry
        for entry in bundle["resources"]
        if entry["kind"] == "KubernetesEndpointResolver"
    ]
    assert {entry["name"] for entry in resolver_resources} == {"myendpoint"}

    replay = _replay(bundle)
    assert replay.complete, replay.reasons
    assert replay.diff == []
    assert replay.signature["resolvers"] == bundle["signature"]["resolvers"]


@pytest.mark.compilertest
def test_built_in_endpoint_alias_round_trips():
    # Only the built-in alias name "endpoint" is defined: the factory must
    # synthesize "kubernetes-endpoint" from it, and replay must reproduce it.
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m1", "default")])
    replay = _replay(bundle)
    assert replay.complete, replay.reasons


@pytest.mark.compilertest
def test_missing_file_is_incomplete_and_never_faked():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    roots = [ClosureRoot("Mapping", "m3", "other")]
    bundle = export_compile_closure(ir, roots)
    assert bundle["completeness"] == "Complete"
    assert bundle["external"]["files"]

    # A file checker that refuses every required file: the replay must stay
    # honest and report Incomplete instead of silently succeeding.
    replay = _replay(bundle, file_checker=lambda path: False)
    assert replay.status == "Incomplete"
    assert any("failed to resolve" in reason for reason in replay.reasons)
    assert replay.diff

    # The same bundle with honest handler-written files reproduces exactly.
    replay_ok = _replay(bundle)
    assert replay_ok.complete, replay_ok.reasons
    assert replay_ok.diff == []


@pytest.mark.compilertest
def test_missing_service_is_a_missing_reference_not_an_unused_resource():
    watt = _watt_snapshot()
    # Drop the Service/Endpoints that m1/m2 rely on.
    watt["Kubernetes"]["service"] = []
    watt["Kubernetes"]["Endpoints"] = []

    result, _ = _compile(watt)
    ir = result["ir"]

    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m1", "default")])

    # The reference is missing (the resolver found no endpoints)...
    assert bundle["completeness"] == "Incomplete"
    missing = bundle["references"]["missing"]
    assert any(
        item["kind"] == "Service" and item["name"] == "k8s-qotm-default"
        for item in missing
    )

    # ...which is distinct from "present but unused".
    unused_names = [item["name"] for item in bundle["references"]["unused"]]
    assert "k8s-qotm-default" not in unused_names

    # Replaying the same incomplete input reproduces the (target-less)
    # closure, but the bundle stays honestly Incomplete because the export
    # itself recorded the missing reference.
    replay = _replay(bundle)
    assert replay.status == "Incomplete"
    assert any("missing Service" in reason for reason in replay.reasons)


@pytest.mark.compilertest
def test_bundle_digest_tampering_is_rejected():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]
    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m1", "default")])

    tampered = copy.deepcopy(bundle)
    tampered["resources"][0]["serialization"] += "\n# tampered\n"

    with pytest.raises(ClosureFormatError, match="digest mismatch"):
        import_compile_closure(tampered)


@pytest.mark.compilertest
def test_path_traversal_is_rejected():
    # Direct helper contract.
    with pytest.raises(ClosureFormatError):
        validate_relative_safety("../etc/passwd")
    with pytest.raises(ClosureFormatError):
        validate_relative_safety("/safe/../../escape")
    with pytest.raises(ClosureFormatError):
        validate_relative_safety("relative/path")

    # Import-time validation rejects a poisoned bundle as well.
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]
    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m3", "other")])
    poisoned = copy.deepcopy(bundle)
    poisoned["external"]["files"][0]["path"] = "/cache/../../etc/passwd"

    with pytest.raises(ClosureFormatError, match="path traversal"):
        import_compile_closure(poisoned)


@pytest.mark.compilertest
def test_limits_are_enforced_on_export_and_import():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    with pytest.raises(ClosureLimitError):
        export_compile_closure(
            ir,
            [ClosureRoot("Mapping", "m1", "default")],
            limits={"max_resources": 1},
        )

    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m1", "default")])
    shrunken = copy.deepcopy(bundle)
    shrunken["limits"]["max_total_bytes"] = 1
    with pytest.raises(ClosureLimitError):
        import_compile_closure(shrunken)


@pytest.mark.compilertest
def test_unknown_roots_and_bad_bundles_are_rejected():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    with pytest.raises(ClosureError):
        export_compile_closure(
            ir, [ClosureRoot("Mapping", "does-not-exist", "default")]
        )

    with pytest.raises(ClosureFormatError):
        import_compile_closure("not-json{")

    with pytest.raises(ClosureFormatError):
        import_compile_closure({"format": "something/else"})


@pytest.mark.compilertest
def test_bundle_is_pure_data_with_compiler_binding():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]
    bundle = export_compile_closure(
        ir,
        [ClosureRoot("Mapping", "m1", "default")],
        input_format="watt-json",
    )

    # JSON round trip: the bundle is data, not executable objects.
    restored = json.loads(json.dumps(bundle))
    assert restored["format"] == BUNDLE_FORMAT
    assert restored["input_format"] == "watt-json"
    assert restored["compiler"]["protocol"] == "v3"
    assert restored["compiler"]["version"]
    assert restored["identity"]["ambassador_id"] == ir.ambassador_id

    # Import accepts serialized JSON as well as a dict, and only parses data.
    serialized = json.dumps(bundle)
    replay_from_json = _replay(serialized)
    assert replay_from_json.complete, replay_from_json.reasons
    assert replay_from_json.diff == []

    # A foreign compiler version forces an honest Incomplete.
    foreign = copy.deepcopy(bundle)
    foreign["compiler"]["version"] = "9.9.9-fake"
    replay = _replay(foreign)
    assert replay.status == "Incomplete"
    assert any("compiler mismatch" in reason for reason in replay.reasons)


@pytest.mark.compilertest
def test_k8s_yaml_input_compiles_and_replays():
    yaml_manifest = """
---
apiVersion: getambassador.io/v3alpha1
kind: Module
metadata:
  name: ambassador
  namespace: default
spec:
  config: {}
---
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: m1
  namespace: default
spec:
  hostname: "*"
  prefix: /a/
  service: qotm
---
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: m2
  namespace: default
spec:
  hostname: "*"
  prefix: /b/
  service: qotm
---
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: unused-mapping
  namespace: default
spec:
  hostname: "*"
  prefix: /z/
  service: other-svc
"""

    result = Compile(logger, yaml_manifest, k8s=True, file_checker=lambda path: True)
    ir = result["ir"]
    assert not _global_errors(ir)

    bundle = export_compile_closure(
        ir,
        [
            ClosureRoot("Mapping", "m1", "default"),
            ClosureRoot("Mapping", "m2", "default"),
        ],
        input_format="k8s-yaml",
    )

    assert bundle["input_format"] == "k8s-yaml"
    assert bundle["completeness"] == "Complete"

    # No external secrets are needed for this plain DNS cluster.
    assert bundle["external"]["secrets"] == []

    replay = import_compile_closure(bundle)
    assert replay.complete, replay.reasons
    assert replay.diff == []

    # The shared cluster identity survives the format change.
    clusters = replay.replay_signature["clusters"]
    assert len(clusters) == 1
    assert "qotm" in next(iter(clusters))


@pytest.mark.compilertest
def test_unknown_dependency_is_marked_opaque():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    # Simulate a future compiler emitting a secret_info key this closure
    # format version does not know how to describe.
    ctx = ir.get_tls_context("upstream-tls")
    assert ctx is not None
    ctx.secret_info["future_material"] = "unmodeled"

    bundle = export_compile_closure(ir, [ClosureRoot("Mapping", "m3", "other")])

    assert bundle["completeness"] == "Incomplete"
    assert any(
        entry["description"].startswith("TLSContext upstream-tls")
        for entry in bundle["external"]["opaque"]
    )

    # Opaque dependencies block a Complete replay as well.
    replay = _replay(bundle)
    assert replay.status == "Incomplete"
    assert any("opaque dependency" in reason for reason in replay.reasons)


@pytest.mark.compilertest
def test_manifest_file_checker_helper_is_path_safe_and_honest():
    present = "/tmp/cache/ns/secrets-decoded/name/abc.crt"
    checker = manifest_file_checker([present])
    assert checker(present)
    assert not checker(present + ".other")

    with pytest.raises(ClosureFormatError):
        manifest_file_checker(["../escape"])


@pytest.mark.compilertest
def test_roots_accept_dict_and_tuple_forms():
    result, _ = _compile(_watt_snapshot())
    ir = result["ir"]

    bundle_a = export_compile_closure(
        ir, [{"kind": "Mapping", "name": "m1", "namespace": "default"}]
    )
    bundle_b = export_compile_closure(ir, [("Mapping", "m1", "default")])
    assert bundle_a["signature"] == bundle_b["signature"]
