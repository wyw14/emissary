# Copyright 2024 Datawire. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
compile_closure -- reviewable, replayable compile closures for explicitly
selected Mapping/Host roots.

The compiler normally consumes an entire snapshot and produces an IR for every
resource in it.  For offline review, auditing, or targeted replay it is useful
to be able to answer a narrower question: given *these* Mappings and/or Hosts,
which resources did the compiler actually use, and which external dependencies
(Secrets, on-disk cert files) did it rely on?

``export_compile_closure`` answers that question by walking the *real* IR object
graph produced by a compile (the same graph Envoy configuration is generated
from), rather than by recursively guessing from resource names.  In
particular:

* dependencies are the object references the IR itself established
  (Mapping -> Group -> Cluster -> Resolver/TLSContext/Service,
  Host -> TLSContext -> Secret);
* cache ownership links (``Cache.link`` owner -> owned) are deliberately *not*
  traversed in reverse, because they would pull in unrelated configuration that
  merely happened to share a cache entry;
* namespaces, ambassador_id, resolver aliases and shared clusters keep their
  real identities;
* referenced-but-absent dependencies are reported as "missing" while present
  but unreached resources are reported separately as "unused".

The exported bundle never contains Secret private keys or other credential
material: only external handles and content digests are recorded.  Replaying a
bundle therefore requires the caller to supply a matching ``SecretHandler``
and/or ``file_checker``.  If the external dependencies cannot be satisfied the
replay is explicitly reported as ``Incomplete``; the replay path never invents
fake secrets and then claims the original output was reproduced.

Bundle import only parses data (JSON + YAML safe loading): it never executes
Python code, never unpickles, and never accesses the network.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import orjson

from .constants import Constants
from .ir.irserviceresolver import is_ip_address
from .utils import EmptySecretHandler, SecretHandler, SecretInfo, parse_yaml
from .VERSION import Commit, Version

if TYPE_CHECKING:  # pragma: no cover
    from .config.acresource import ACResource
    from .ir.ir import IR
    from .ir.ircluster import IRCluster
    from .ir.irhost import IRHost
    from .ir.irresource import IRResource
    from .ir.irserviceresolver import IRServiceResolver
    from .ir.irtlscontext import IRTLSContext

#############################################################################
# Constants
#############################################################################

BUNDLE_FORMAT = "ambassador.compile-closure/1"
PROTOCOL_VERSION = "v3"

INPUT_K8S_YAML = "k8s-yaml"
INPUT_WATT_JSON = "watt-json"
_INPUT_FORMATS = (INPUT_K8S_YAML, INPUT_WATT_JSON)

# Completeness / dependency statuses.
COMPLETE = "Complete"
INCOMPLETE = "Incomplete"

PRESENT = "Present"
MISSING = "Missing"
OPAQUE = "Opaque"

ROOT_KINDS = ("Mapping", "TCPMapping", "Host")

# Secret elements that can carry credential material.
_SECRET_ELEMENTS = ("tls_crt", "tls_key", "user_key", "root_crt")

# Secret name references carried inside an IRTLSContext.secret_info.
_SECRET_NAME_KEYS = ("secret", "ca_secret", "crl_secret")

# On-disk file references carried inside an IRTLSContext.secret_info after the
# IR has resolved its secrets.
_SECRET_FILE_KEYS = (
    "cert_chain_file",
    "private_key_file",
    "cacert_chain_file",
    "crl_file",
)

# Keys that describe compiler bookkeeping rather than configuration semantics.
# They vary across compiles (rkeys, locations, cache bookkeeping) and must not
# participate in field-by-field replay comparison.
_VOLATILE_KEYS = frozenset(
    {
        "rkey",
        "_rkey",
        "location",
        "_referenced_by",
        "_cache_key",
        "_active",
        "_errored",
        "serialization",
        "apiVersion",
        "logger",
        "ir",
    }
)

# Flat (normalized) resource keys that are not part of a k8s CRD spec.
_FLAT_RESERVED_KEYS = frozenset(
    {"apiVersion", "kind", "name", "namespace", "generation", "metadata_labels"}
)

# Default safety limits.  These are deliberately generous; callers may override
# them via the ``limits`` parameter, but never raise them past the hard caps.
DEFAULT_LIMITS = {
    "max_resources": 1000,
    "max_resource_bytes": 1 * 1024 * 1024,
    "max_total_bytes": 16 * 1024 * 1024,
    "max_manifest_entries": 2000,
}

_HARD_CAPS = {
    "max_resources": 10_000,
    "max_resource_bytes": 16 * 1024 * 1024,
    "max_total_bytes": 256 * 1024 * 1024,
    "max_manifest_entries": 20_000,
}


#############################################################################
# Errors and public data types
#############################################################################


class ClosureError(Exception):
    """Base class for closure export/import failures."""


class ClosureLimitError(ClosureError):
    """Raised when a bundle exceeds a configured resource/byte limit."""


class ClosureFormatError(ClosureError):
    """Raised when a bundle cannot be parsed or fails validation."""


@dataclass(frozen=True)
class ClosureRoot:
    """
    An explicitly selected closure root.

    :param kind: "Mapping", "TCPMapping" or "Host"
    :param name: resource name
    :param namespace: resource namespace; defaults to the IR's ambassador namespace
    """

    kind: str
    name: str
    namespace: Optional[str] = None

    @classmethod
    def coerce(
        cls, root: "ClosureRoot | Dict[str, Any] | Sequence[Any]"
    ) -> "ClosureRoot":
        if isinstance(root, ClosureRoot):
            return root
        if isinstance(root, dict):
            return cls(
                kind=str(root["kind"]),
                name=str(root["name"]),
                namespace=root.get("namespace"),
            )
        if isinstance(root, (tuple, list)):
            if len(root) == 2:
                return cls(kind=str(root[0]), name=str(root[1]))
            if len(root) == 3:
                return cls(kind=str(root[0]), name=str(root[1]), namespace=root[2])
        raise ClosureError(f"invalid closure root: {root!r}")

    @property
    def identity(self) -> Tuple[str, str, str]:
        return (self.kind, self.name, self.namespace or "")


@dataclass
class ReplayResult:
    """
    The outcome of importing and replaying a closure bundle.

    ``status`` is ``Complete`` only when every external dependency was
    satisfied and the field-by-field comparison against the original closure
    signature found no differences.  A ``Incomplete`` result always carries
    human-readable ``reasons`` and must never be treated as a faithful
    reproduction of the original compile.
    """

    status: str
    reasons: List[str]
    diff: List[str]
    signature: Dict[str, Any]
    replay_signature: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    bundle: Dict[str, Any]

    @property
    def complete(self) -> bool:
        return self.status == COMPLETE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "complete": self.complete,
            "reasons": list(self.reasons),
            "diff": list(self.diff),
            "signature": self.signature,
            "replay_signature": self.replay_signature,
        }


#############################################################################
# Small helpers
#############################################################################


def sha256_digest(data: str) -> str:
    """Return a stable content digest for a string."""
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_relative_safety(path: str) -> str:
    """
    Reject paths that try to escape their base directory via ".." segments or
    that are not absolute filesystem paths.  Returns the validated path.
    """
    if not isinstance(path, str) or not path:
        raise ClosureFormatError(f"invalid file path: {path!r}")

    # Normalize both POSIX and Windows separators so "..\\.." cannot sneak
    # past a POSIX-only check.
    normalized = path.replace("\\", "/")

    if ".." in normalized.split("/"):
        raise ClosureFormatError(f"path traversal is not allowed: {path!r}")

    # A NUL byte is never legal in a filesystem path and is a classic injection
    # marker.
    if "\x00" in path:
        raise ClosureFormatError(f"NUL byte in file path: {path!r}")

    # Accept both native absolute paths (e.g. Windows drive paths) and
    # POSIX-rooted container paths such as /ambassador/....
    if not os.path.isabs(path) and not normalized.startswith("/"):
        raise ClosureFormatError(f"external file paths must be absolute: {path!r}")

    return path


def _canonical(value: Any) -> Any:
    """
    Return a plain-data, volatility-stripped copy of an IR value suitable for
    field-by-field comparison.  Nested IRResource dicts are recursed into;
    rkeys, locations and cache bookkeeping are dropped.
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            if key in _VOLATILE_KEYS:
                continue
            canonical = _canonical(val)
            if canonical is not None:
                # Stringify keys so the signature is stable across JSON
                # round trips (JSON object keys are always strings).
                out[str(key)] = canonical
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _diff_paths(a: Any, b: Any, path: str = "") -> List[str]:
    """Recursively describe field differences between two canonical values."""
    diffs: List[str] = []

    if isinstance(a, dict) and isinstance(b, dict):
        # key=str: JSON round trips turn integer mapping keys into strings, so
        # the two sides may not share a single key type.
        for key in sorted(set(a) | set(b), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in a:
                diffs.append(f"{child}: only in replay ({b[key]!r})")
            elif key not in b:
                diffs.append(f"{child}: only in original ({a[key]!r})")
            else:
                diffs.extend(_diff_paths(a[key], b[key], child))
        return diffs

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        for idx, (av, bv) in enumerate(zip(a, b)):
            diffs.extend(_diff_paths(av, bv, f"{path}[{idx}]"))
        return diffs

    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


#############################################################################
# Closure collection from the real IR object graph
#############################################################################


class _ClosureCollector:
    def __init__(
        self, ir: "IR", roots: Sequence[ClosureRoot], logger: logging.Logger
    ) -> None:
        self.ir = ir
        self.roots = roots
        self.logger = logger

        self.groups: Dict[str, IRResource] = {}
        self.mappings: Dict[str, IRResource] = {}
        self.clusters: Dict[str, IRCluster] = {}
        self.resolvers: Dict[str, IRServiceResolver] = {}
        self.contexts: Dict[str, IRTLSContext] = {}
        self.hosts: Dict[str, IRHost] = {}
        self.services: Dict[str, ACResource] = {}

        # (secret name, namespace) -> manifest entry accumulator
        self.secret_refs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # file path -> manifest entry accumulator
        self.file_refs: Dict[str, Dict[str, Any]] = {}

        self.edges: List[Dict[str, Any]] = []
        self.missing: List[Dict[str, Any]] = []
        self.opaque: List[Dict[str, Any]] = []

        self._secret_materials = self._collect_secret_materials()

    #
    # IR-side secret material, used to record content digests without ever
    # placing the material itself in the bundle.
    #
    def _collect_secret_materials(self) -> Dict[Tuple[str, str], Dict[str, str]]:
        materials: Dict[Tuple[str, str], Dict[str, str]] = {}

        for secret_info in self.ir.secret_info.values():
            elements = {
                element: getattr(secret_info, element)
                for element in _SECRET_ELEMENTS
                if getattr(secret_info, element, None)
            }
            if elements:
                materials[(secret_info.name, secret_info.namespace)] = elements

        for saved in self.ir.saved_secrets.values():
            if not saved.cert_data:
                continue
            elements = {
                element: saved.cert_data.get(element)
                for element in _SECRET_ELEMENTS
                if saved.cert_data.get(element)
            }
            if elements:
                materials.setdefault((saved.secret_name, saved.namespace), elements)

        return materials

    #
    # Root resolution
    #
    def resolve_roots(self) -> None:
        for root in self.roots:
            if root.kind in ("Mapping", "TCPMapping"):
                self._resolve_mapping_root(root)
            elif root.kind == "Host":
                self._resolve_host_root(root)
            else:
                raise ClosureError(
                    f"unsupported root kind {root.kind!r}; expected one of {ROOT_KINDS}"
                )

    def _resolve_mapping_root(self, root: ClosureRoot) -> None:
        namespace = root.namespace or self.ir.ambassador_namespace
        found = False

        for group in self.ir.groups.values():
            candidates = list(group.mappings)
            shadows = getattr(group, "shadows", [])
            candidates.extend(shadows)

            for mapping in candidates:
                if mapping.name != root.name or mapping.namespace != namespace:
                    continue
                found = True
                self.groups[group.group_id] = group
                self.mappings[mapping.cache_key] = mapping
                self._add_edge(
                    "root",
                    root.identity,
                    "mapping",
                    (mapping.kind, mapping.name, mapping.namespace),
                )
                self._walk_mapping(mapping)

        if not found:
            raise ClosureError(
                f"root {root.kind} {root.name} in namespace {namespace} is not part of the "
                "compiled IR (unknown or inactive)"
            )

    def _resolve_host_root(self, root: ClosureRoot) -> None:
        namespace = root.namespace or self.ir.ambassador_namespace
        host = self.ir.hosts.get(root.name)

        if host is None or host.namespace != namespace:
            raise ClosureError(
                f"root Host {root.name} in namespace {namespace} is not part of the compiled IR"
            )

        self.hosts[host.name] = host
        self._add_edge(
            "root", root.identity, "host", (host.kind, host.name, host.namespace)
        )
        self._walk_host(host)

    #
    # Graph walks -- only real IR object references are followed here.
    #
    def _walk_mapping(self, mapping: "IRResource") -> None:
        cluster = getattr(mapping, "cluster", None)

        if cluster is None:
            # host_redirect mappings deliberately have no cluster.
            self._add_edge(
                "mapping",
                (mapping.kind, mapping.name, mapping.namespace),
                "cluster",
                None,
            )
            return

        self.clusters[cluster.name] = cluster
        self._add_edge(
            "mapping",
            (mapping.kind, mapping.name, mapping.namespace),
            "cluster",
            (cluster.kind, cluster.name, cluster.namespace),
        )
        self._walk_cluster(cluster)

    def _walk_cluster(self, cluster: "IRCluster") -> None:
        # Resolver: the cluster records the resolver name it was created with;
        # the default resolver comes from the Ambassador module.  This keeps a
        # resolver alias (e.g. "endpoint" vs "kubernetes-endpoint") as its real
        # identity instead of normalizing the name away.
        resolver_name = cluster._resolver or self.ir.ambassador_module.get(
            "resolver", "kubernetes-service"
        )
        resolver = self.ir.get_resolver(resolver_name)

        if resolver is None:
            self._record_missing(
                "Resolver",
                resolver_name,
                cluster.namespace,
                required_by=f"cluster {cluster.name}",
                reason="cluster references a resolver that is not present in the IR",
            )
        else:
            self.resolvers[resolver.name] = resolver
            self._add_edge(
                "cluster",
                (cluster.kind, cluster.name, cluster.namespace),
                "resolver",
                (resolver.kind, resolver.name, resolver.namespace),
            )
            self._walk_service_dependency(cluster, resolver)

        tls_context = cluster.get("tls_context")
        if tls_context and not tls_context.get("_ambassador_enabled", False):
            self.contexts[tls_context.name] = tls_context
            self._add_edge(
                "cluster",
                (cluster.kind, cluster.name, cluster.namespace),
                "tlscontext",
                (tls_context.kind, tls_context.name, tls_context.namespace),
            )
            self._walk_tls_context(tls_context, required_by=f"cluster {cluster.name}")

    def _walk_service_dependency(
        self, cluster: "IRCluster", resolver: "IRServiceResolver"
    ) -> None:
        """
        Determine the Service/endpoints resource the resolver actually used for
        this cluster, mirroring IRServiceResolver.resolve() rather than guessing
        from the service name string.
        """
        hostname = cluster._hostname

        # IP-address upstreams resolve without a Service resource at all.
        # Use the same predicate the resolver itself uses.
        if is_ip_address(hostname):
            return

        if resolver.kind == "KubernetesServiceResolver":
            # DNS-name resolution; no Service/endpoints resource is consumed.
            return

        if resolver.kind == "KubernetesEndpointResolver":
            service_name, namespace = resolver.parse_service(
                self.ir, hostname, cluster._namespace
            )
            key = f"k8s-{service_name}-{namespace}"
        elif resolver.kind == "ConsulResolver":
            datacenter = resolver.get("datacenter")
            if not datacenter:
                self._record_opaque(
                    f"consul resolver {resolver.name} has no datacenter",
                    required_by=f"cluster {cluster.name}",
                )
                return
            key = f"consul-{hostname}-{datacenter}"
        else:
            self._record_opaque(
                f"unknown resolver kind {resolver.kind!r} for resolver {resolver.name}",
                required_by=f"cluster {cluster.name}",
            )
            return

        service = self.ir.services.get(key)

        if service is None:
            self._record_missing(
                "Service",
                key,
                cluster._namespace,
                required_by=f"cluster {cluster.name}",
                reason=f"resolver {resolver.name} found no Service/endpoints for {key}",
            )
            return

        self.services[key] = service
        self._add_edge(
            "cluster",
            (cluster.kind, cluster.name, cluster.namespace),
            "service",
            (service.kind, service.name, service.namespace),
        )

    def _walk_tls_context(self, ctx: "IRTLSContext", required_by: str) -> None:
        secret_info = ctx.get("secret_info") or {}

        # Named Secret references.  Namespace handling must match
        # IRTLSContext.resolve_secret() exactly, including the "name.namespace"
        # cross-namespace syntax and the secret_namespacing toggle.
        namespacing = ctx.lookup(
            "secret_namespacing", True, default_key="tls_secret_namespacing"
        )

        for key in _SECRET_NAME_KEYS:
            secret_name = secret_info.get(key)
            if not secret_name:
                continue

            name, namespace = secret_name, ctx.namespace
            if "." in secret_name and namespacing:
                name, namespace = secret_name.rsplit(".", 1)

            self._record_secret(name, namespace, required_by)

        # Files the SecretHandler materialized on disk.
        for key in _SECRET_FILE_KEYS:
            path = secret_info.get(key)
            if path:
                self._record_file(path, required_by)

        # Anything else in secret_info is something this version of the closure
        # format does not know how to describe: mark it Opaque rather than
        # silently dropping it.
        known = set(_SECRET_NAME_KEYS) | set(_SECRET_FILE_KEYS)
        unknown = set(secret_info.keys()) - known
        for key in sorted(unknown):
            self._record_opaque(
                f"TLSContext {ctx.name} has unmodeled secret_info key {key!r}",
                required_by=required_by,
            )

    def _walk_host(self, host: "IRHost") -> None:
        ctx = getattr(host, "context", None)
        if ctx and not ctx.get("_ambassador_enabled", False):
            self.contexts[ctx.name] = ctx
            self._add_edge(
                "host",
                (host.kind, host.name, host.namespace),
                "tlscontext",
                (ctx.kind, ctx.name, ctx.namespace),
            )
            self._walk_tls_context(ctx, required_by=f"host {host.name}")

        tls_secret = host.get("tlsSecret") or {}
        if tls_secret.get("name"):
            namespace = tls_secret.get("namespace") or host.namespace
            self._record_secret(
                tls_secret["name"], namespace, required_by=f"host {host.name}"
            )

        acme = host.get("acmeProvider") or {}
        private_key = acme.get("privateKeySecret") or {}
        if private_key.get("name"):
            # IRHost.resolve() defaults the namespace to the Host namespace.
            self._record_secret(
                private_key["name"],
                host.namespace,
                required_by=f"host {host.name} (acme)",
            )

    #
    # External dependency accumulators
    #
    def _record_secret(self, name: str, namespace: str, required_by: str) -> None:
        key = (name, namespace)
        entry = self.secret_refs.setdefault(
            key,
            {
                "handle": f"k8s:v1/Secret/{namespace}/{name}",
                "name": name,
                "namespace": namespace,
                "digests": {},
                "status": MISSING,
                "required_by": set(),
            },
        )
        entry["required_by"].add(required_by)

        material = self._secret_materials.get(key)
        if material:
            entry["status"] = PRESENT
            for element, value in material.items():
                entry["digests"][element] = sha256_digest(value)

    def _record_file(self, path: str, required_by: str) -> None:
        validate_relative_safety(path)
        entry = self.file_refs.setdefault(
            path,
            {"path": path, "status": MISSING, "required_by": set()},
        )
        entry["required_by"].add(required_by)
        entry["status"] = PRESENT if self.ir.file_checker(path) else MISSING

    def _record_missing(
        self, kind: str, name: str, namespace: str, required_by: str, reason: str
    ) -> None:
        self.missing.append(
            {
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "required_by": required_by,
                "reason": reason,
            }
        )

    def _record_opaque(self, description: str, required_by: str) -> None:
        self.opaque.append({"description": description, "required_by": required_by})

    def _add_edge(
        self,
        from_kind: str,
        from_id: Any,
        to_kind: str,
        to_id: Any,
    ) -> None:
        self.edges.append({"from": [from_kind, from_id], "to": [to_kind, to_id]})


#############################################################################
# Signature construction (field-by-field comparison basis)
#############################################################################


def _canonical_cluster(cluster: "IRCluster") -> Dict[str, Any]:
    data = _canonical(cluster.as_dict())
    tls_context = cluster.get("tls_context")
    if tls_context:
        data["tls_context"] = tls_context.name
    else:
        data["tls_context"] = None
    data["resolver"] = cluster._resolver or "kubernetes-service"
    return data


def _canonical_context(ctx: "IRTLSContext") -> Dict[str, Any]:
    data = _canonical(ctx.as_dict())
    secret_info = ctx.get("secret_info") or {}
    reduced: Dict[str, Any] = {}

    for key in _SECRET_NAME_KEYS:
        if key in secret_info:
            reduced[key] = secret_info[key]
    for key in _SECRET_FILE_KEYS:
        reduced[f"{key}_present"] = bool(secret_info.get(key))

    data["secret_info"] = reduced
    return data


def _canonical_host(host: "IRHost") -> Dict[str, Any]:
    data = _canonical(host.as_dict())
    ctx = getattr(host, "context", None)
    data["context"] = ctx.name if ctx else None
    return data


def _canonical_group(group: "IRResource") -> Dict[str, Any]:
    data = _canonical(group.as_dict())

    for section in ("mappings", "shadows"):
        entries = data.get(section)
        if not entries:
            continue
        for mapping in entries:
            cluster = mapping.get("cluster")
            mapping["cluster"] = (
                cluster.get("name") if isinstance(cluster, dict) else None
            )

    return data


def _canonical_service(service: "ACResource") -> Dict[str, Any]:
    return _canonical(service.as_dict())


def build_closure_signature(ir: "IR", roots: Sequence[ClosureRoot]) -> Dict[str, Any]:
    """
    Build the canonical, environment-independent description of the selected
    closure in an IR.  The same function is run against both the original IR
    and the replayed IR, so exact equality of the two results constitutes a
    field-by-field verification of the selected output closure.
    """
    logger = ir.logger
    collector = _ClosureCollector(ir, roots, logger)
    collector.resolve_roots()

    module_signature = _canonical(ir.ambassador_module.as_dict())

    # service_port is a derived global: the IR flips it to HTTPS when *any*
    # termination context exists in the whole compile, including contexts
    # outside the selected closure.  Re-derive it from the closure's own
    # contexts so the signature describes the closure rather than the
    # original snapshot's unrelated resources.
    has_termination_context = any(
        bool(ctx.get("hosts")) for ctx in collector.contexts.values()
    )
    module_signature["service_port"] = (
        Constants.SERVICE_PORT_HTTPS
        if has_termination_context
        else Constants.SERVICE_PORT_HTTP
    )

    signature: Dict[str, Any] = {
        "identity": {
            "ambassador_id": ir.ambassador_id,
            "ambassador_namespace": ir.ambassador_namespace,
            "ambassador_nodename": ir.ambassador_nodename,
        },
        "roots": sorted(list(root.identity) for root in roots),
        "module": module_signature,
        "hosts": {
            name: _canonical_host(host)
            for name, host in sorted(collector.hosts.items())
        },
        "groups": [
            _canonical_group(collector.groups[group_id])
            for group_id in sorted(collector.groups)
        ],
        "clusters": {
            name: _canonical_cluster(cluster)
            for name, cluster in sorted(collector.clusters.items())
        },
        "resolvers": {
            name: {
                "kind": resolver.kind,
                "name": resolver.name,
                "namespace": resolver.namespace,
                "resolve_with": resolver.get("resolve_with"),
                "datacenter": resolver.get("datacenter"),
            }
            for name, resolver in sorted(collector.resolvers.items())
        },
        "tls_contexts": {
            name: _canonical_context(ctx)
            for name, ctx in sorted(collector.contexts.items())
        },
        "services": {
            key: _canonical_service(service)
            for key, service in sorted(collector.services.items())
        },
    }

    return signature


#############################################################################
# Export
#############################################################################


def _aconf_match(
    store: Dict[str, "ACResource"], name: str, namespace: str
) -> Optional["ACResource"]:
    for resource in store.values():
        if resource.name == name and resource.namespace == namespace:
            return resource
    return None


def _check_limits(
    limits: Dict[str, int], resource_count: int, total_bytes: int, manifest_count: int
) -> None:
    if resource_count > limits["max_resources"]:
        raise ClosureLimitError(
            f"closure has {resource_count} resources, limit is {limits['max_resources']}"
        )
    if total_bytes > limits["max_total_bytes"]:
        raise ClosureLimitError(
            f"closure bundle is {total_bytes} bytes, limit is {limits['max_total_bytes']}"
        )
    if manifest_count > limits["max_manifest_entries"]:
        raise ClosureLimitError(
            f"closure has {manifest_count} external dependencies, limit is "
            f"{limits['max_manifest_entries']}"
        )


def export_compile_closure(
    ir: "IR",
    roots: Sequence["ClosureRoot | Dict[str, Any] | Sequence[Any]"],
    *,
    input_format: str = INPUT_K8S_YAML,
    logger: Optional[logging.Logger] = None,
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Export a reviewable input bundle plus external dependency manifest for the
    closure reachable from the explicitly selected ``roots`` in an already
    compiled ``ir``.

    :param ir: a compiled :class:`ambassador.IR` (the fixed, resolved configuration)
    :param roots: selected Mapping/TCPMapping/Host roots (ClosureRoot, dict or tuple)
    :param input_format: the format of the original compiler input
    :param logger: optional logger
    :param limits: override the default safety limits
    :returns: a JSON-serializable bundle dictionary
    """
    logger = logger or ir.logger

    if input_format not in _INPUT_FORMATS:
        raise ClosureError(
            f"unknown input_format {input_format!r}; expected one of {_INPUT_FORMATS}"
        )

    effective_limits = dict(DEFAULT_LIMITS)
    if limits:
        for key, value in limits.items():
            if key not in DEFAULT_LIMITS:
                raise ClosureError(f"unknown limit {key!r}")
            if value > _HARD_CAPS[key]:
                raise ClosureLimitError(
                    f"limit {key}={value} exceeds hard cap {_HARD_CAPS[key]}"
                )
            effective_limits[key] = value

    coerced_roots: List[ClosureRoot] = []
    for raw_root in roots:
        root = ClosureRoot.coerce(raw_root)
        if root.kind not in ROOT_KINDS:
            raise ClosureError(
                f"unsupported root kind {root.kind!r}; expected one of {ROOT_KINDS}"
            )
        coerced_roots.append(
            ClosureRoot(root.kind, root.name, root.namespace or ir.ambassador_namespace)
        )

    if not coerced_roots:
        raise ClosureError("at least one closure root must be selected")

    collector = _ClosureCollector(ir, coerced_roots, logger)
    collector.resolve_roots()

    resources, included_identities, has_legacy_tls = _build_resources(ir, collector)

    unused = _collect_unused(ir, included_identities)

    resource_entries: List[Dict[str, Any]] = []
    total_bytes = 0

    for resource in resources:
        serialization = resource.serialization or ""
        byte_size = len(serialization.encode("utf-8"))

        if byte_size > effective_limits["max_resource_bytes"]:
            raise ClosureLimitError(
                f"resource {resource.kind} {resource.name}.{resource.namespace} is {byte_size} "
                f"bytes, limit is {effective_limits['max_resource_bytes']}"
            )

        total_bytes += byte_size
        resource_entries.append(
            {
                "rkey": resource.rkey,
                "kind": resource.kind,
                "name": resource.name,
                "namespace": resource.namespace,
                "apiVersion": resource.apiVersion,
                "origin": "user",
                "serialization": serialization,
                "digest": sha256_digest(serialization),
                "byte_size": byte_size,
            }
        )

    manifest_entries = (
        len(collector.secret_refs) + len(collector.file_refs) + len(collector.opaque)
    )
    _check_limits(
        effective_limits, len(resource_entries), total_bytes, manifest_entries
    )

    incomplete_reasons: List[str] = []

    for entry in collector.missing:
        incomplete_reasons.append(
            f"missing {entry['kind']} {entry['name']}.{entry['namespace']} "
            f"(required by {entry['required_by']}): {entry['reason']}"
        )

    for entry in collector.secret_refs.values():
        if entry["status"] == MISSING:
            incomplete_reasons.append(
                f"missing Secret {entry['name']}.{entry['namespace']} "
                f"(required by {sorted(entry['required_by'])})"
            )

    for entry in collector.file_refs.values():
        if entry["status"] == MISSING:
            incomplete_reasons.append(
                f"missing file {entry['path']} (required by {sorted(entry['required_by'])})"
            )

    for entry in collector.opaque:
        incomplete_reasons.append(
            f"opaque dependency: {entry['description']} (required by {entry['required_by']})"
        )

    bundle: Dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "compiler": {
            "package": "ambassador",
            "version": Version,
            "commit": Commit,
            "protocol": PROTOCOL_VERSION,
        },
        "input_format": input_format,
        "identity": {
            "ambassador_id": ir.ambassador_id,
            "ambassador_namespace": ir.ambassador_namespace,
            "ambassador_nodename": ir.ambassador_nodename,
        },
        "roots": [
            {"kind": root.kind, "name": root.name, "namespace": root.namespace}
            for root in coerced_roots
        ],
        "resources": resource_entries,
        "external": {
            "secrets": [
                {
                    "handle": entry["handle"],
                    "name": entry["name"],
                    "namespace": entry["namespace"],
                    "status": entry["status"],
                    "digests": dict(sorted(entry["digests"].items())),
                    "required_by": sorted(entry["required_by"]),
                }
                for entry in sorted(
                    collector.secret_refs.values(), key=lambda e: e["handle"]
                )
            ],
            "files": [
                {
                    "path": entry["path"],
                    "status": entry["status"],
                    "required_by": sorted(entry["required_by"]),
                }
                for entry in sorted(
                    collector.file_refs.values(), key=lambda e: e["path"]
                )
            ],
            "opaque": list(collector.opaque),
        },
        "references": {
            "edges": collector.edges,
            "missing": collector.missing,
            "unused": unused,
        },
        "signature": build_closure_signature(ir, coerced_roots),
        "limits": effective_limits,
        "completeness": INCOMPLETE if incomplete_reasons else COMPLETE,
        "incomplete_reasons": incomplete_reasons,
        "legacy_tls_module": has_legacy_tls,
    }

    return bundle


def _build_resources(
    ir: "IR", collector: _ClosureCollector
) -> Tuple[List["ACResource"], Set[Tuple[str, str, str]], bool]:
    """
    Map every closure entity back to its source ACResource in the fixed
    aconf.  Secret ACResources are deliberately excluded: credential material
    never travels in the bundle, it is represented by external handles and
    digests instead.
    """
    aconf = ir.aconf
    chosen: Dict[str, ACResource] = {}
    identities: Set[Tuple[str, str, str]] = set()

    def take(resource: Optional["ACResource"]) -> None:
        if resource is None:
            return
        if resource.kind == "Secret":
            # Never bundle Secret material, no matter how it was reached.
            return
        chosen[resource.rkey] = resource
        identities.add((resource.kind, resource.name, resource.namespace))

    # The Ambassador module is always part of the closure (defaults, resolver
    # defaults, labels, ...).
    ambassador_module = aconf.get_module("ambassador")
    take(ambassador_module)

    # Mapping roots.
    mapping_store = aconf.get_config("mappings") or {}
    tcp_store = aconf.get_config("tcpmappings") or {}
    for mapping in collector.mappings.values():
        take(_aconf_match(mapping_store, mapping.name, mapping.namespace))
        take(_aconf_match(tcp_store, mapping.name, mapping.namespace))
        identities.add((mapping.kind, mapping.name, mapping.namespace))

    # Host roots.
    host_store = aconf.get_config("hosts") or {}
    for host in collector.hosts.values():
        take(_aconf_match(host_store, host.name, host.namespace))
        identities.add((host.kind, host.name, host.namespace))

    # Resolvers: resolver aliases and synthesized default resolvers have no
    # source ACResource; the compiler recreates them on replay, so they are
    # recorded as identities without a bundle resource.
    resolver_store = aconf.get_config("resolvers") or {}
    for resolver in collector.resolvers.values():
        source = _aconf_match(resolver_store, resolver.name, resolver.namespace)
        take(source)
        identities.add((resolver.kind, resolver.name, resolver.namespace))

    # TLS contexts, including implicit ones synthesized by a Host.
    context_store = aconf.get_config("tls_contexts") or {}
    has_legacy_tls = False
    for ctx in collector.contexts.values():
        source = _aconf_match(context_store, ctx.name, ctx.namespace)
        take(source)
        identities.add((ctx.kind, ctx.name, ctx.namespace))
        if ctx.get("_legacy"):
            has_legacy_tls = True

    if has_legacy_tls:
        tls_module = aconf.get_module("tls")
        take(tls_module)

    # Services / endpoints.
    service_store = aconf.get_config("service") or {}
    for key, service in collector.services.items():
        take(service_store.get(key))
        identities.add((service.kind, service.name, service.namespace))

    return list(chosen.values()), identities, has_legacy_tls


def _collect_unused(
    ir: "IR", included: Set[Tuple[str, str, str]]
) -> List[Dict[str, str]]:
    """
    Enumerate resources present in the fixed configuration but not reached by
    the selected closure.  Identities only -- never serializations -- so an
    unused resource cannot leak content into the bundle.
    """
    store_names = (
        "mappings",
        "tcpmappings",
        "hosts",
        "tls_contexts",
        "resolvers",
        "service",
        "secrets",
        "modules",
    )

    unused: List[Dict[str, str]] = []

    for store_name in store_names:
        store = ir.aconf.get_config(store_name) or {}
        for resource in store.values():
            identity = (resource.kind, resource.name, resource.namespace)
            if identity not in included:
                unused.append(
                    {
                        "kind": resource.kind,
                        "name": resource.name,
                        "namespace": resource.namespace,
                    }
                )

    return sorted(
        unused, key=lambda item: (item["kind"], item["namespace"], item["name"])
    )


#############################################################################
# Replay support: honest SecretHandler / file checkers
#############################################################################


class ManifestSecretHandler(SecretHandler):
    """
    A SecretHandler for closure replay that serves only caller-provided
    material.  It never invents fake credentials: an unknown secret returns
    None, exactly like a cluster that does not hold the requested Secret.
    """

    def __init__(
        self,
        logger: logging.Logger,
        material: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._tempdir = tempfile.TemporaryDirectory(
            prefix="closure-secret-", suffix="-cache"
        )
        super().__init__(
            logger, self._tempdir.name, self._tempdir.name, "closure-replay"
        )
        # Keyed "name.namespace"; values hold already-decoded PEM elements.
        self.material = dict(material or {})

    def load_secret(
        self, resource: "IRResource", secret_name: str, namespace: str
    ) -> Optional[SecretInfo]:
        data = self.material.get(f"{secret_name}.{namespace}")

        if data is None:
            return None

        return SecretInfo(
            secret_name,
            namespace,
            data.get("secret_type", "kubernetes.io/tls"),
            tls_crt=data.get("tls_crt"),
            tls_key=data.get("tls_key"),
            user_key=data.get("user_key"),
            root_crt=data.get("root_crt"),
            decode_b64=False,
        )


def manifest_file_checker(present_paths: Sequence[str]) -> "Callable[[str], bool]":
    """
    Build a file checker that honestly reports the caller-provided set of
    available files.  Every other path is reported missing, so replay cannot
    accidentally succeed against ambient files on the replay host.
    """
    present = set(present_paths)
    for path in present:
        validate_relative_safety(path)

    def checker(path: str) -> bool:
        return path in present

    return checker


class _HandlerResourceShim:
    """Minimal resource stand-in for SecretHandler.load_secret calls."""

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name


#############################################################################
# Bundle -> WATT snapshot reconstruction (pure data transformation)
#############################################################################


def _wrap_as_k8s_object(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of ResourceManager's normalized flat Ambassador object."""
    spec = {key: value for key, value in flat.items() if key not in _FLAT_RESERVED_KEYS}

    metadata: Dict[str, Any] = {
        "name": flat["name"],
        "namespace": flat["namespace"],
        "labels": flat.get("metadata_labels") or {},
    }

    if "generation" in flat:
        metadata["generation"] = flat["generation"]

    return {
        "apiVersion": flat["apiVersion"],
        "kind": flat["kind"],
        "metadata": metadata,
        "spec": spec,
    }


def _reconstruct_service(
    flat: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Rebuild core v1 Service/Endpoints objects (or a Consul snapshot block)
    from the normalized Ambassador Service resource stored in the IR.  This is
    a deterministic, data-only transformation of bundle content.
    """
    name = flat["name"]
    namespace = flat["namespace"]
    endpoints = flat.get("endpoints") or {}

    if flat.get("datacenter"):
        datacenter = flat["datacenter"]
        consul_endpoints = []
        for entries in endpoints.values():
            for endpoint in entries:
                consul_endpoints.append(
                    {"Address": endpoint["ip"], "Port": endpoint["port"]}
                )
        rkey = f"consul-{name}-{datacenter}"
        consul = {
            "Endpoints": {
                rkey: {
                    "Service": name,
                    "Id": datacenter,
                    "Endpoints": consul_endpoints,
                }
            }
        }
        return [], [], consul

    service_ports: List[Dict[str, Any]] = []
    endpoint_ports: List[Dict[str, Any]] = []
    addresses: Set[str] = set()

    for src_port in sorted(endpoints, key=lambda p: int(p)):
        entries = endpoints[src_port]
        target_port = entries[0]["port"] if entries else int(src_port)

        service_ports.append(
            {
                "name": f"port-{src_port}",
                "port": int(src_port),
                "targetPort": target_port,
                "protocol": "TCP",
            }
        )
        endpoint_ports.append(
            {"name": f"port-{src_port}", "port": target_port, "protocol": "TCP"}
        )
        for endpoint in entries:
            addresses.add(endpoint["ip"])

    metadata = {"name": name, "namespace": namespace}
    labels = flat.get("metadata_labels") or {}

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {**dict(metadata), "labels": dict(labels)},
        "spec": {"ports": service_ports, "selector": {}},
    }
    endpoints_object = {
        "apiVersion": "v1",
        "kind": "Endpoints",
        "metadata": dict(metadata),
        "subsets": [
            {
                "addresses": [{"ip": ip} for ip in sorted(addresses)],
                "ports": endpoint_ports,
            }
        ],
    }

    return [service], [endpoints_object], None


def _build_watt_snapshot(bundle: Dict[str, Any]) -> Dict[str, Any]:
    kubernetes: Dict[str, List[Dict[str, Any]]] = {}
    consul: Dict[str, Any] = {}

    for entry in bundle["resources"]:
        documents = parse_yaml(entry["serialization"])
        if not documents:
            continue
        flat = documents[0]

        if flat["kind"] == "Service":
            services, endpoint_objects, consul_block = _reconstruct_service(flat)
            kubernetes.setdefault("service", []).extend(services)
            kubernetes.setdefault("Endpoints", []).extend(endpoint_objects)
            if consul_block:
                consul.setdefault("Endpoints", {}).update(consul_block["Endpoints"])
            continue

        kubernetes.setdefault(flat["kind"], []).append(_wrap_as_k8s_object(flat))

    snapshot: Dict[str, Any] = {"Kubernetes": kubernetes}
    if consul:
        snapshot["Consul"] = consul
    return snapshot


#############################################################################
# Import / replay
#############################################################################


def _validate_bundle(bundle: Any) -> Dict[str, Any]:
    if not isinstance(bundle, dict):
        raise ClosureFormatError("closure bundle must be a JSON object")

    if bundle.get("format") != BUNDLE_FORMAT:
        raise ClosureFormatError(
            f"unsupported closure format {bundle.get('format')!r}; expected {BUNDLE_FORMAT!r}"
        )

    resources = bundle.get("resources")
    if not isinstance(resources, list):
        raise ClosureFormatError("bundle 'resources' must be a list")

    limits = bundle.get("limits") or DEFAULT_LIMITS

    total_bytes = 0
    for entry in resources:
        serialization = entry.get("serialization", "")
        if not isinstance(serialization, str):
            raise ClosureFormatError(
                f"resource {entry.get('rkey')!r} has no string serialization"
            )

        byte_size = len(serialization.encode("utf-8"))
        total_bytes += byte_size

        if byte_size > limits["max_resource_bytes"]:
            raise ClosureLimitError(
                f"resource {entry.get('rkey')!r} exceeds max_resource_bytes"
            )
        if sha256_digest(serialization) != entry.get("digest"):
            raise ClosureFormatError(
                f"digest mismatch for resource {entry.get('rkey')!r}; "
                "the source resource changed after the bundle was exported"
            )

    if len(resources) > limits["max_resources"]:
        raise ClosureLimitError("bundle exceeds max_resources")
    if total_bytes > limits["max_total_bytes"]:
        raise ClosureLimitError("bundle exceeds max_total_bytes")

    external = bundle.get("external") or {}
    manifest_count = (
        len(external.get("secrets") or [])
        + len(external.get("files") or [])
        + len(external.get("opaque") or [])
    )
    if manifest_count > limits["max_manifest_entries"]:
        raise ClosureLimitError("bundle exceeds max_manifest_entries")

    for file_entry in external.get("files") or []:
        validate_relative_safety(file_entry["path"])

    return bundle


def _check_secret_satisfaction(
    bundle: Dict[str, Any],
    secret_handler: SecretHandler,
    logger: logging.Logger,
) -> List[str]:
    reasons: List[str] = []

    for entry in bundle["external"]["secrets"]:
        shim = _HandlerResourceShim("closure", entry["name"])
        secret_info = secret_handler.load_secret(
            shim, entry["name"], entry["namespace"]
        )

        if secret_info is None:
            reasons.append(
                f"external Secret {entry['name']}.{entry['namespace']} was not provided by the "
                "SecretHandler"
            )
            continue

        for element, expected_digest in entry["digests"].items():
            value = getattr(secret_info, element, None)
            if value is None:
                reasons.append(
                    f"external Secret {entry['name']}.{entry['namespace']} is missing element "
                    f"{element}"
                )
            elif sha256_digest(value) != expected_digest:
                reasons.append(
                    f"external Secret {entry['name']}.{entry['namespace']} element {element} "
                    "digest does not match the bundle manifest"
                )

    return reasons


def import_compile_closure(
    bundle: "Dict[str, Any] | str | bytes",
    *,
    secret_handler: Optional[SecretHandler] = None,
    file_checker: Optional["Callable[[str], bool]"] = None,
    logger: Optional[logging.Logger] = None,
) -> ReplayResult:
    """
    Parse a closure bundle, verify its bindings and replay it through the real
    :func:`ambassador.compile.Compile` pipeline.

    Import only parses data (JSON via orjson, YAML via the safe loader): no
    Python code is executed from the bundle and no network access occurs.
    Secrets are never read from the bundle; ``secret_handler`` and
    ``file_checker`` mediate every external dependency.  When any dependency
    cannot be satisfied the result is :data:`INCOMPLETE` with explicit reasons;
    the replay never substitutes fake credentials and claims success.

    :param bundle: bundle dict or JSON serialization produced by
        :func:`export_compile_closure`
    :param secret_handler: caller-owned handler providing the real Secrets
    :param file_checker: caller-owned checker for on-disk cert files
    :param logger: optional logger
    :returns: :class:`ReplayResult` with the comparison outcome and diffs
    """
    logger = logger or logging.getLogger("ambassador.compile_closure")

    if isinstance(bundle, (str, bytes)):
        try:
            parsed = orjson.loads(bundle)
        except orjson.JSONDecodeError as exc:
            raise ClosureFormatError(f"bundle is not valid JSON: {exc}") from exc
    elif isinstance(bundle, dict):
        parsed = bundle
    else:
        raise ClosureFormatError("bundle must be a dict or JSON string/bytes")

    _validate_bundle(parsed)

    roots = [
        ClosureRoot(entry["kind"], entry["name"], entry["namespace"])
        for entry in parsed["roots"]
    ]

    reasons: List[str] = []

    # An export that was Incomplete (missing references, opaque dependencies)
    # can never become a fully reproducible replay: carry its findings forward.
    for reason in parsed.get("incomplete_reasons") or []:
        reasons.append(f"export-time finding: {reason}")

    # Compiler/protocol binding: a different compiler cannot be guaranteed to
    # reproduce the original output.
    compiler = parsed.get("compiler") or {}
    if (
        compiler.get("version") != Version
        or compiler.get("protocol") != PROTOCOL_VERSION
    ):
        reasons.append(
            f"compiler mismatch: bundle was exported with "
            f"{compiler.get('version')} (protocol {compiler.get('protocol')}), replay uses "
            f"{Version} (protocol {PROTOCOL_VERSION})"
        )

    external = parsed["external"]

    # Honest defaults: an absent SecretHandler returns None for every secret
    # rather than fabricating one; an absent file checker marks every required
    # file missing.  NullSecretHandler (which invents fake keys) is never used.
    effective_handler = secret_handler or EmptySecretHandler(
        logger, "", "", "closure-replay"
    )

    required_files = [entry["path"] for entry in external.get("files") or []]
    if file_checker is not None:
        effective_checker = file_checker
    elif not required_files:

        def effective_checker(path: str) -> bool:
            return True

    else:
        # Honest default: a file counts as present only when the replay
        # SecretHandler actually materialized it inside its own cache
        # directory.  Paths outside that cache (or any path when no handler
        # was given) are reported missing -- ambient files on the replay host
        # must never make an unsatisfied replay look successful.
        def effective_checker(path: str) -> bool:
            cache_dir = getattr(effective_handler, "cache_dir", "") or ""
            if not cache_dir:
                return False
            try:
                common = os.path.commonpath(
                    [os.path.abspath(cache_dir), os.path.abspath(path)]
                )
            except ValueError:
                return False
            if common != os.path.abspath(cache_dir):
                return False
            return os.path.isfile(path)

    reasons.extend(_check_secret_satisfaction(parsed, effective_handler, logger))

    for entry in external.get("opaque") or []:
        reasons.append(f"opaque dependency cannot be replayed: {entry['description']}")

    # Note: file dependencies are not pre-checked against the manifest paths.
    # Those paths describe the original compile environment; during replay the
    # SecretHandler materializes files under its own cache directory, and the
    # effective file_checker is invoked by the compiler with those new paths.
    # Satisfaction is therefore established after the replay compile, by
    # confirming every originally present TLS context survived resolution.

    # Data-only reconstruction of a WATT snapshot, then the real compiler.
    snapshot = _build_watt_snapshot(parsed)

    # Import locally to avoid a module-import cycle.
    from .compile import Compile

    result = Compile(
        logger,
        orjson.dumps(snapshot).decode("utf-8"),
        secret_handler=effective_handler,
        file_checker=effective_checker,
        k8s=False,
    )

    replay_ir = result["ir"]

    # A root can disappear from the replay IR when an external dependency
    # could not be satisfied (e.g. a Host whose TLS secret was not provided).
    # Report that explicitly instead of failing the whole import.
    try:
        replay_signature = build_closure_signature(replay_ir, roots)
    except ClosureError as exc:
        replay_signature = None
        reasons.append(f"replay could not reconstruct the selected closure: {exc}")

    original_signature = parsed["signature"]

    if replay_signature is not None:
        diff = _diff_paths(original_signature, replay_signature)

        original_contexts = set(original_signature.get("tls_contexts", {}))
        replay_contexts = set(replay_signature.get("tls_contexts", {}))
        for ctx_name in sorted(original_contexts - replay_contexts):
            reasons.append(
                f"TLSContext {ctx_name} failed to resolve during replay "
                "(a required Secret or file dependency was not available)"
            )
    else:
        diff = ["replay closure was not constructed; see reasons"]

    if diff:
        reasons.append(
            f"replayed closure differs from the original in {len(diff)} field(s)"
        )

    status = COMPLETE if not reasons else INCOMPLETE

    return ReplayResult(
        status=status,
        reasons=reasons,
        diff=diff,
        signature=original_signature,
        replay_signature=replay_signature,
        result=result,
        bundle=parsed,
    )
