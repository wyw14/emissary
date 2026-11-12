# Copyright 2026 Datawire. All rights reserved.
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
plan_host_basis -- finite, snapshot-bound Host subset planning.

Given an immutable snapshot of the Listener, HTTP mapping group, Module and TLS
dependency world plus a finite set of Host candidates (each with a non-negative
integer cost), plan_host_basis searches for the minimum-cost Host subset that
satisfies a finite set of declared binding obligations.

A required obligation names a Listener identity and an HTTP mapping group
identity and demands that at least one selected Host simultaneously passes the
real IRListener.matches_host and IRHost.matches_httpgroup predicates. A
forbidden obligation demands that no selected Host establishes that binding.

The planner never reimplements selector or matching semantics: namespace
SELF/ALL, hostBinding.matchLabels, the mappingSelector/hostname combination,
the DISABLE_STRICT_LABEL_SELECTORS compatibility switch and the synthetic
mapping behavior are all exercised through the real compiler objects produced
by isolated Compile runs. Unknown dynamic dependencies are never treated as
match success: when the fixed caller snapshot lacks SecretHandler or
file_checker inputs, planning stops with status Incomplete.

The search is a bounded branch-and-bound over the finite candidate space using
a covering lower bound for pruning (no item-by-item greedy approximation).
Every structurally complete selection is additionally checked by an isolated
real Compile; TLS/filter-chain combination conflicts therefore make the whole
selection infeasible. An oracle failure only ever excludes the exact selection
that was verified; it is never extrapolated into pruning other combinations.

The result is a planning artifact only. It performs no Kubernetes writes and no
deletions, never edits candidate hostnames or labels, and its conclusions hold
only for the finite obligations and the fixed snapshot supplied by the caller.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

from .compile import Compile
from .ir.irhost import IRHost
from .ir.irhttpmappinggroup import IRHTTPMappingGroup
from .ir.irlistener import IRListener
from .ir.irutils import disable_strict_selectors
from .utils import SecretHandler, SecretInfo

# Hard safety limits. The problem is intentionally finite; callers that need
# larger worlds must make separate, explicit planning requests.
MAX_CANDIDATES = 16
MAX_OBLIGATIONS = 24
DEFAULT_MAX_EVALUATIONS = 256
MAX_EVALUATIONS_CAP = 1024
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_SAMPLE_ENTRIES = 10

STATUS_MINIMUM = "Minimum"
STATUS_BEST_KNOWN = "BestKnown"
STATUS_INFEASIBLE = "Infeasible"
STATUS_INCOMPLETE = "Incomplete"

_SCOPE_NOTE = (
    "Conclusions hold only for the finite obligations and the fixed caller "
    "snapshot supplied; future requests or future Hosts are not guaranteed safe."
)


@dataclass(frozen=True)
class HostCandidate:
    """
    A single selectable Host.

    candidate_id is the caller-chosen stable identifier. manifest is exactly
    one Kubernetes Host document (YAML text); the planner only ever includes or
    excludes it wholesale and never rewrites hostnames, labels or any other
    field. cost is a non-negative integer and must_keep forces inclusion.
    """

    candidate_id: str
    manifest: str
    cost: int
    must_keep: bool = False


@dataclass(frozen=True)
class BindingObligation:
    """
    A declared binding obligation.

    listener and group are stable "namespace/name" references into the fixed
    snapshot. With forbid=False, at least one selected Host must establish the
    listener-to-group binding; with forbid=True, no selected Host may establish
    it.
    """

    obligation_id: str
    listener: str
    group: str
    forbid: bool = False


@dataclass(frozen=True)
class HostBasisSnapshot:
    """
    The immutable planning input.

    base_manifests holds the fixed Listener/Mapping/Module/TLS resources.
    secrets maps "name.namespace" keys to the SecretInfo values available to
    the SecretHandler; files maps absolute paths to their fixed file_checker
    results. max_evaluations bounds the number of real Compile oracle calls.
    """

    base_manifests: str
    candidates: Sequence[HostCandidate]
    obligations: Sequence[BindingObligation]
    secrets: Mapping[str, SecretInfo]
    files: Mapping[str, bool]
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS


class _SnapshotSecretHandler(SecretHandler):
    """
    SecretHandler backed entirely by the fixed caller snapshot.

    A load_secret request for anything absent from the snapshot is recorded in
    missing_secrets and fails closed (returns None); nothing online is ever
    consulted.
    """

    def __init__(
        self,
        logger: logging.Logger,
        secrets: Mapping[str, SecretInfo],
    ) -> None:
        self._tempdir_source = tempfile.TemporaryDirectory(
            prefix="plan-host-basis-", suffix="-source"
        )
        self._tempdir_cache = tempfile.TemporaryDirectory(
            prefix="plan-host-basis-", suffix="-cache"
        )
        super().__init__(
            logger,
            self._tempdir_source.name,
            self._tempdir_cache.name,
            "plan-host-basis-snapshot",
        )
        self._secrets = dict(secrets)
        self.missing_secrets: List[str] = []

    def load_secret(
        self, resource: Any, secret_name: str, namespace: str
    ) -> Optional[SecretInfo]:
        key = f"{secret_name}.{namespace}"
        secret_info = self._secrets.get(key)
        if secret_info is None:
            self.logger.debug(
                "plan_host_basis: snapshot has no secret %s (requested by %s %s)",
                key,
                getattr(resource, "kind", "?"),
                getattr(resource, "name", "?"),
            )
            self.missing_secrets.append(key)
            return None
        return secret_info

    def cleanup(self) -> None:
        self._tempdir_source.cleanup()
        self._tempdir_cache.cleanup()


class _SnapshotFileChecker:
    """
    file_checker backed entirely by fixed snapshot data.

    Paths explicitly enumerated in the caller's files map are answered from
    that map (a TLSContext pointing at an on-disk file is a real snapshot
    dependency). Paths underneath the snapshot SecretHandler cache directory
    are materialized by the handler from the fixed SecretInfo snapshot, so
    their existence is a deterministic consequence of the snapshot and is
    checked directly. Any other path is an unknown, potentially changing
    dependency: it is recorded in missing_files and fails closed.
    """

    # IR.__init__ reads file_checker.__name__ for a debug log line.
    __name__ = "plan_host_basis_snapshot_file_checker"

    def __init__(self, files: Mapping[str, bool], cache_root: str) -> None:
        self._files = dict(files)
        self._cache_root = os.path.abspath(cache_root)
        self.missing_files: List[str] = []

    def __call__(self, path: str) -> bool:
        if path in self._files:
            return bool(self._files[path])
        # Paths the snapshot SecretHandler itself materialized are derived
        # inputs, not changing external dependencies.
        if os.path.abspath(path).startswith(self._cache_root + os.sep):
            return os.path.isfile(path)
        self.missing_files.append(path)
        return False


@dataclass
class _OracleResult:
    ok: bool
    new_errors: List[str]
    missing_secrets: List[str]
    missing_files: List[str]
    raised: Optional[str] = None


@dataclass
class _Node:
    # Position into the cost-ordered candidate permutation.
    next_pos: int
    selected_mask: int
    cost: int
    count: int
    lower_bound: int


class _HostBasisPlanner:
    def __init__(self, snapshot: HostBasisSnapshot, logger: logging.Logger) -> None:
        self.snapshot = snapshot
        self.logger = logger
        self.strict_selectors_disabled = disable_strict_selectors()

        # Filled in during planning.
        self._candidate_docs: List[Dict[str, Any]] = []
        self._host_refs: List[str] = []
        self._inactive: List[str] = []
        self._listener_index: Dict[str, IRListener] = {}
        self._group_index: Dict[str, IRHTTPMappingGroup] = {}
        self._required: List[BindingObligation] = []
        self._forbidden: List[BindingObligation] = []
        self._cover_mask: List[int] = []
        self._witness_sets: List[int] = []
        self._excluded_mask = 0
        self._forced_mask = 0
        self._order: List[int] = []
        self._baseline_errors: Counter[str] = Counter()
        self._oracle_cache: Dict[int, _OracleResult] = {}
        self._oracle_ir: Dict[int, Any] = {}

        self.evaluations = 0
        self.pruned = 0
        self.nodes_visited = 0
        self.exhausted = False

        # Incumbent state.
        self._inc_mask: Optional[int] = None
        self._inc_key: Optional[Tuple[int, int, Tuple[str, ...]]] = None

        # Failure samples for the artifact (capped).
        self.failures: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # Validation and digest
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        snap = self.snapshot

        if not 1 <= len(snap.candidates) <= MAX_CANDIDATES:
            raise ValueError(f"candidate count must be between 1 and {MAX_CANDIDATES}")
        if not 1 <= len(snap.obligations) <= MAX_OBLIGATIONS:
            raise ValueError(
                f"obligation count must be between 1 and {MAX_OBLIGATIONS}"
            )
        if not 1 <= snap.max_evaluations <= MAX_EVALUATIONS_CAP:
            raise ValueError(
                f"max_evaluations must be between 1 and {MAX_EVALUATIONS_CAP}"
            )

        candidate_ids: Set[str] = set()
        host_identities: Set[Tuple[str, str]] = set()
        for candidate in snap.candidates:
            if not candidate.candidate_id:
                raise ValueError("candidate_id must be non-empty")
            if candidate.candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate_id {candidate.candidate_id}")
            candidate_ids.add(candidate.candidate_id)

            if (
                not isinstance(candidate.cost, int)
                or isinstance(candidate.cost, bool)
                or candidate.cost < 0
            ):
                raise ValueError(
                    f"candidate {candidate.candidate_id} cost must be a "
                    "non-negative integer"
                )

            docs = [doc for doc in yaml.safe_load_all(candidate.manifest) if doc]
            if len(docs) != 1:
                raise ValueError(
                    f"candidate {candidate.candidate_id} manifest must contain "
                    "exactly one document"
                )
            doc = docs[0]
            if doc.get("kind") != "Host":
                raise ValueError(
                    f"candidate {candidate.candidate_id} manifest must be kind Host"
                )
            metadata = doc.get("metadata") or {}
            name = metadata.get("name")
            namespace = metadata.get("namespace")
            if not name or not namespace:
                raise ValueError(
                    f"candidate {candidate.candidate_id} Host requires metadata "
                    "name and namespace"
                )
            identity = (namespace, name)
            if identity in host_identities:
                raise ValueError(
                    f"duplicate Host identity {namespace}/{name} across candidates"
                )
            host_identities.add(identity)
            self._candidate_docs.append(doc)
            self._host_refs.append(f"{namespace}/{name}")

        obligation_ids: Set[str] = set()
        for obligation in snap.obligations:
            if not obligation.obligation_id:
                raise ValueError("obligation_id must be non-empty")
            if obligation.obligation_id in obligation_ids:
                raise ValueError(f"duplicate obligation_id {obligation.obligation_id}")
            obligation_ids.add(obligation.obligation_id)
            _split_ref(obligation.obligation_id, obligation.listener, "listener")
            _split_ref(obligation.obligation_id, obligation.group, "group")

    def _digest(self) -> str:
        snap = self.snapshot
        payload = {
            "base_manifests": snap.base_manifests,
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "manifest": c.manifest,
                    "cost": c.cost,
                    "must_keep": c.must_keep,
                }
                for c in snap.candidates
            ],
            "obligations": [
                {
                    "obligation_id": o.obligation_id,
                    "listener": o.listener,
                    "group": o.group,
                    "forbid": o.forbid,
                }
                for o in snap.obligations
            ],
            "secrets": {
                key: secret_info.to_dict() for key, secret_info in snap.secrets.items()
            },
            "files": dict(snap.files),
            "max_evaluations": snap.max_evaluations,
            "strict_selectors_disabled": self.strict_selectors_disabled,
        }
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return "sha256:" + hashlib.sha256(canonical).hexdigest()

    # ------------------------------------------------------------------
    # Isolated real Compile oracle
    # ------------------------------------------------------------------

    def _compile(
        self, yaml_text: str
    ) -> Tuple[Any, _SnapshotSecretHandler, _SnapshotFileChecker]:
        secret_handler = _SnapshotSecretHandler(self.logger, self.snapshot.secrets)
        file_checker = _SnapshotFileChecker(
            self.snapshot.files, secret_handler.cache_dir
        )
        # cache=None gives every run a fresh NullCache: caller caches can never
        # be observed or mutated.
        result = Compile(
            self.logger,
            yaml_text,
            cache=None,
            file_checker=file_checker,
            secret_handler=secret_handler,
            k8s=True,
        )
        return result["ir"], secret_handler, file_checker

    def _yaml_for(self, selected_mask: int) -> str:
        parts = [self.snapshot.base_manifests]
        for idx, candidate in enumerate(self.snapshot.candidates):
            if selected_mask & (1 << idx):
                parts.append(candidate.manifest)
        return _join_manifests(parts)

    @staticmethod
    def _error_signature(ir: Any) -> Counter[str]:
        signature: Counter[str] = Counter()
        for entries in ir.aconf.errors.values():
            for entry in entries:
                signature[json.dumps(entry, sort_keys=True, default=str)] += 1
        return signature

    def _oracle(self, selected_mask: int) -> _OracleResult:
        cached = self._oracle_cache.get(selected_mask)
        if cached is not None:
            return cached

        self.evaluations += 1
        handler: Optional[_SnapshotSecretHandler] = None
        try:
            yaml_text = self._yaml_for(selected_mask)
            ir, handler, checker = self._compile(yaml_text)
            missing_secrets = list(handler.missing_secrets)
            missing_files = list(checker.missing_files)
            errors = self._error_signature(ir)
            new_errors = sorted((errors - self._baseline_errors).elements())
            result = _OracleResult(
                ok=not new_errors and not missing_secrets and not missing_files,
                new_errors=new_errors,
                missing_secrets=missing_secrets,
                missing_files=missing_files,
            )
            self._oracle_ir[selected_mask] = ir
        except Exception as exc:  # compiler crash fails this selection only
            result = _OracleResult(
                ok=False,
                new_errors=[],
                missing_secrets=[],
                missing_files=[],
                raised=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if handler is not None:
                handler.cleanup()

        self._oracle_cache[selected_mask] = result
        if not result.ok and len(self.failures) < MAX_SAMPLE_ENTRIES:
            self.failures.append(
                {
                    "selected_hosts": self._mask_ids(selected_mask),
                    "new_errors": result.new_errors[:MAX_SAMPLE_ENTRIES],
                    "missing_secrets": result.missing_secrets[:MAX_SAMPLE_ENTRIES],
                    "missing_files": result.missing_files[:MAX_SAMPLE_ENTRIES],
                    "raised": result.raised,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Structural model built from a real all-candidate Compile
    # ------------------------------------------------------------------

    def _build_model(self) -> Optional[Dict[str, Any]]:
        # Baseline: fixed snapshot without any candidate Hosts.
        baseline_handler: Optional[_SnapshotSecretHandler] = None
        baseline_checker: Optional[_SnapshotFileChecker] = None
        try:
            baseline_ir, baseline_handler, baseline_checker = self._compile(
                _join_manifests([self.snapshot.base_manifests])
            )
        except Exception as exc:
            return self._incomplete(
                [],
                [],
                f"baseline compile raised {type(exc).__name__}: {exc}",
            )
        finally:
            if baseline_handler is not None:
                baseline_handler.cleanup()

        assert baseline_checker is not None
        missing = list(baseline_handler.missing_secrets)
        missing_files = list(baseline_checker.missing_files)
        if missing or missing_files:
            return self._incomplete(missing, missing_files, None)
        self._baseline_errors = self._error_signature(baseline_ir)

        # Analysis compile: every candidate at once. The resulting IR carries
        # the real IRListener/IRHost/IRHTTPMappingGroup objects whose predicate
        # methods define all matching semantics.
        all_mask = (1 << len(self.snapshot.candidates)) - 1
        analysis_handler: Optional[_SnapshotSecretHandler] = None
        analysis_checker: Optional[_SnapshotFileChecker] = None
        try:
            analysis_ir, analysis_handler, analysis_checker = self._compile(
                self._yaml_for(all_mask)
            )
        except Exception as exc:
            # Without the analysis IR no predicate evidence exists; never claim
            # infeasibility, return an unproven BestKnown instead.
            self.warnings.append(f"analysis compile raised {type(exc).__name__}: {exc}")
            return self._artifact(
                STATUS_BEST_KNOWN,
                None,
                None,
                None,
            )
        finally:
            if analysis_handler is not None:
                missing = list(analysis_handler.missing_secrets)
                missing_files = list(analysis_checker.missing_files)
                analysis_handler.cleanup()

        assert analysis_checker is not None
        if missing or missing_files:
            return self._incomplete(missing, missing_files, None)

        for listener in analysis_ir.listeners.values():
            self._listener_index[f"{listener.namespace}/{listener.name}"] = listener

        for group in analysis_ir.groups.values():
            if isinstance(group, IRHTTPMappingGroup):
                self._group_index[f"{group.namespace}/{group.name}"] = group

        # Resolve candidate Hosts to real IRHost objects. A candidate whose Host
        # did not become active establishes nothing and is never treated as a
        # successful match for an unknown dependency.
        hosts_by_identity: Dict[Tuple[str, str], IRHost] = {
            (host.namespace, host.name): host for host in analysis_ir.get_hosts()
        }
        candidate_hosts: List[Optional[IRHost]] = []
        for idx, doc in enumerate(self._candidate_docs):
            metadata = doc["metadata"]
            identity = (metadata["namespace"], metadata["name"])
            host = hosts_by_identity.get(identity)
            if host is None:
                self._inactive.append(self.snapshot.candidates[idx].candidate_id)
            candidate_hosts.append(host)

        for obligation in self.snapshot.obligations:
            listener_ref = obligation.listener
            group_ref = obligation.group
            listener = self._listener_index.get(listener_ref)
            if listener is None:
                raise ValueError(
                    f"obligation {obligation.obligation_id} references unknown "
                    f"listener {listener_ref}"
                )
            group = self._group_index.get(group_ref)
            if group is None:
                raise ValueError(
                    f"obligation {obligation.obligation_id} references unknown "
                    f"HTTP mapping group {group_ref}"
                )
            if obligation.forbid:
                self._forbidden.append(obligation)
            else:
                self._required.append(obligation)

        n = len(self.snapshot.candidates)
        self._cover_mask = [0 for _ in range(n)]

        # Required obligations: real predicate evidence only.
        for j, obligation in enumerate(self._required):
            listener = self._listener_index[obligation.listener]
            group = self._group_index[obligation.group]
            witnesses = 0
            for idx, host in enumerate(candidate_hosts):
                if host is None:
                    continue
                if listener.matches_host(host) and host.matches_httpgroup(group):
                    witnesses |= 1 << idx
                    self._cover_mask[idx] |= 1 << j
            self._witness_sets.append(witnesses)

        # Forbidden obligations: any candidate establishing the binding is
        # hard-excluded from the search.
        self._excluded_mask = 0
        for obligation in self._forbidden:
            listener = self._listener_index[obligation.listener]
            group = self._group_index[obligation.group]
            for idx, host in enumerate(candidate_hosts):
                if host is None:
                    continue
                if listener.matches_host(host) and host.matches_httpgroup(group):
                    self._excluded_mask |= 1 << idx

        self._forced_mask = 0
        for idx, candidate in enumerate(self.snapshot.candidates):
            if candidate.must_keep:
                self._forced_mask |= 1 << idx

        # Cost-ordered, identity-stable exploration permutation.
        self._order = sorted(
            range(n),
            key=lambda i: (
                self.snapshot.candidates[i].cost,
                self._host_refs[i],
                self.snapshot.candidates[i].candidate_id,
            ),
        )
        return None

    # ------------------------------------------------------------------
    # Branch and bound
    # ------------------------------------------------------------------

    def _node_lower_bound(
        self, selected_mask: int, next_pos: int
    ) -> Optional[Tuple[int, int]]:
        """
        Admissible covering lower bound.

        For every still-uncovered obligation take the cheapest undecided
        candidate able to witness it; the maximum of those minima (not the
        sum, which would double-count shared candidates) is a valid lower
        bound on the additional cost. Returns None when the branch cannot
        cover its remaining obligations at all.
        """
        covered_obligations = 0
        for idx in range(len(self.snapshot.candidates)):
            if selected_mask & (1 << idx):
                covered_obligations |= self._cover_mask[idx]

        required_bits = (1 << len(self._required)) - 1
        uncovered = required_bits & ~covered_obligations
        current_count = bin(selected_mask).count("1")

        if not uncovered:
            return 0, current_count

        remaining = 0
        for pos in range(next_pos, len(self._order)):
            remaining |= 1 << self._order[pos]
        # Forced Hosts are already selected and their coverage is accounted
        # for above; excluded Hosts can never be part of any completion.
        remaining &= ~(self._forced_mask | self._excluded_mask)

        min_added_cost = 0
        for j in range(len(self._required)):
            if not (uncovered & (1 << j)):
                continue
            options = self._witness_sets[j] & remaining
            if not options:
                return None
            cheapest = min(
                self.snapshot.candidates[idx].cost for idx in self._iter_bits(options)
            )
            min_added_cost = max(min_added_cost, cheapest)

        current_cost = self._mask_cost(selected_mask)
        # Completing the cover needs at least one extra Host.
        return current_cost + min_added_cost, current_count + 1

    def _mask_cost(self, mask: int) -> int:
        return sum(self.snapshot.candidates[idx].cost for idx in self._iter_bits(mask))

    def _mask_ids(self, mask: int) -> List[str]:
        return [
            self.snapshot.candidates[idx].candidate_id for idx in self._iter_bits(mask)
        ]

    def _mask_key(self, mask: int) -> Tuple[int, int, Tuple[str, ...]]:
        refs = tuple(sorted(self._host_refs[idx] for idx in self._iter_bits(mask)))
        return self._mask_cost(mask), bin(mask).count("1"), refs

    @staticmethod
    def _iter_bits(mask: int):
        bit = 0
        while mask:
            if mask & 1:
                yield bit
            mask >>= 1
            bit += 1

    def _consider_incumbent(self, mask: int) -> bool:
        """
        Verify a structurally complete selection with the real Compile oracle
        and, when it compiles cleanly, fold it into the incumbent. Returns the
        oracle verdict.
        """
        result = self._oracle(mask)
        if result.missing_secrets or result.missing_files:
            # Snapshot incompleteness discovered mid-search is global.
            raise _Incomplete(result.missing_secrets, result.missing_files)
        if not result.ok:
            return False
        key = self._mask_key(mask)
        if self._inc_key is None or key < self._inc_key:
            self._inc_key = key
            self._inc_mask = mask
        return True

    def search(self) -> Dict[str, Any]:
        n = len(self.snapshot.candidates)

        # Structural impossibilities that are proofs in the finite world.
        if self._forced_mask & self._excluded_mask:
            self.warnings.append(
                "a must-keep candidate establishes a forbidden binding"
            )
            return self._artifact(STATUS_INFEASIBLE, None, None, None)

        required_bits = (1 << len(self._required)) - 1
        for j in range(len(self._required)):
            if not (self._witness_sets[j] & ~self._excluded_mask):
                self.warnings.append(
                    f"obligation {self._required[j].obligation_id} has no "
                    "candidate able to establish its binding"
                )
                return self._artifact(STATUS_INFEASIBLE, None, None, None)

        root = self._node_lower_bound(self._forced_mask, 0)
        # root cannot be None here: every required obligation has at least one
        # non-excluded witness and every candidate is still undecided.
        assert root is not None
        stack: List[_Node] = [
            _Node(
                0,
                self._forced_mask,
                self._mask_cost(self._forced_mask),
                bin(self._forced_mask).count("1"),
                root[0],
            )
        ]
        frontier_lower_bounds: List[int] = [root[0]]

        while stack:
            if self.evaluations >= self.snapshot.max_evaluations:
                self.exhausted = True
                break

            node = stack.pop()
            frontier_lower_bounds.pop()
            self.nodes_visited += 1

            bound = self._node_lower_bound(node.selected_mask, node.next_pos)
            if bound is None:
                self.pruned += 1
                continue
            lb_cost, lb_count = bound
            node.lower_bound = lb_cost

            if self._inc_key is not None:
                inc_cost, inc_count, _ = self._inc_key
                if (lb_cost, lb_count) > (inc_cost, inc_count):
                    self.pruned += 1
                    continue

            # Coverage complete? Verify the selection through the real oracle.
            covered = 0
            for idx in self._iter_bits(node.selected_mask):
                covered |= self._cover_mask[idx]
            if covered == required_bits:
                ok = self._consider_incumbent(node.selected_mask)
                if ok:
                    # Every superset costs at least as much and uses at least
                    # as many Hosts, so it is structurally dominated and never
                    # needs its own oracle call. This is objective dominance,
                    # not extrapolation of an oracle verdict.
                    self.pruned += 1
                    continue
                if self.evaluations >= self.snapshot.max_evaluations:
                    self.exhausted = True
                    frontier_lower_bounds.append(lb_cost)
                    break

            if node.next_pos >= n:
                continue

            candidate_idx = self._order[node.next_pos]
            next_pos = node.next_pos + 1

            if self._forced_mask & (1 << candidate_idx):
                # Must-keep candidates are already selected; they get a single
                # continuation rather than duplicate include/exclude branches.
                forced_bound = self._node_lower_bound(node.selected_mask, next_pos)
                if forced_bound is not None:
                    stack.append(
                        _Node(
                            next_pos,
                            node.selected_mask,
                            node.cost,
                            node.count,
                            forced_bound[0],
                        )
                    )
                    frontier_lower_bounds.append(forced_bound[0])
                continue

            # Exclude branch first (popped second, explored later).
            exclude_node = _Node(
                next_pos,
                node.selected_mask,
                node.cost,
                node.count,
                lb_cost,
            )
            stack.append(exclude_node)
            frontier_lower_bounds.append(lb_cost)

            if not (self._excluded_mask & (1 << candidate_idx)):
                include_mask = node.selected_mask | (1 << candidate_idx)
                include_bound = self._node_lower_bound(include_mask, next_pos)
                if include_bound is not None:
                    include_node = _Node(
                        next_pos,
                        include_mask,
                        self._mask_cost(include_mask),
                        node.count + 1,
                        include_bound[0],
                    )
                    stack.append(include_node)
                    frontier_lower_bounds.append(include_bound[0])

        if self.exhausted:
            # Every open frontier node still carries an admissible LB, so the
            # cheapest of them bounds the unexplored optimum; the root LB is a
            # sound fallback when the frontier was drained exactly at budget.
            lower_bound = (
                min(frontier_lower_bounds) if frontier_lower_bounds else root[0]
            )
        elif self._inc_mask is not None:
            # The whole bounded tree was explored or soundly pruned: the
            # verified incumbent is provably optimal, so bounds coincide.
            lower_bound = self._mask_cost(self._inc_mask)
        else:
            lower_bound = None

        if self._inc_mask is not None:
            upper_bound = self._mask_cost(self._inc_mask)
            if lower_bound is not None and lower_bound == upper_bound:
                status = STATUS_MINIMUM
            else:
                status = STATUS_BEST_KNOWN
        else:
            upper_bound = None
            status = STATUS_BEST_KNOWN if self.exhausted else STATUS_INFEASIBLE

        return self._artifact(
            status,
            self._inc_mask,
            lower_bound,
            upper_bound,
        )

    # ------------------------------------------------------------------
    # Artifact assembly
    # ------------------------------------------------------------------

    def _incomplete(
        self,
        missing_secrets: List[str],
        missing_files: List[str],
        extra: Optional[str],
    ) -> Dict[str, Any]:
        if extra:
            self.warnings.append(extra)
        return self._artifact(
            STATUS_INCOMPLETE,
            None,
            None,
            None,
            missing_secrets=missing_secrets,
            missing_files=missing_files,
        )

    def _witnesses_for(self, mask: Optional[int]) -> List[Dict[str, Any]]:
        if mask is None or mask not in self._oracle_ir:
            return [
                {
                    "obligation_id": obligation.obligation_id,
                    "listener": obligation.listener,
                    "group": obligation.group,
                    "forbid": obligation.forbid,
                    "witnesses": [],
                }
                for obligation in self.snapshot.obligations
            ]

        ir = self._oracle_ir[mask]
        hosts_by_identity = {
            (host.namespace, host.name): host for host in ir.get_hosts()
        }
        listeners_by_ref = {
            f"{listener.namespace}/{listener.name}": listener
            for listener in ir.listeners.values()
        }
        groups_by_ref = {
            f"{group.namespace}/{group.name}": group
            for group in ir.groups.values()
            if isinstance(group, IRHTTPMappingGroup)
        }
        details: List[Dict[str, Any]] = []
        for obligation in self.snapshot.obligations:
            listener = listeners_by_ref.get(obligation.listener)
            group = groups_by_ref.get(obligation.group)
            witness_ids: List[str] = []
            for idx, doc in enumerate(self._candidate_docs):
                if not (mask & (1 << idx)):
                    continue
                metadata = doc["metadata"]
                host = hosts_by_identity.get((metadata["namespace"], metadata["name"]))
                if host is None or listener is None or group is None:
                    continue
                if listener.matches_host(host) and host.matches_httpgroup(group):
                    witness_ids.append(self.snapshot.candidates[idx].candidate_id)
            if obligation.forbid:
                # Structural construction guarantees this.
                assert not witness_ids
            details.append(
                {
                    "obligation_id": obligation.obligation_id,
                    "listener": obligation.listener,
                    "group": obligation.group,
                    "forbid": obligation.forbid,
                    "witnesses": sorted(witness_ids),
                }
            )
        return details

    def _artifact(
        self,
        status: str,
        mask: Optional[int],
        lower_bound: Optional[int],
        upper_bound: Optional[int],
        missing_secrets: Optional[List[str]] = None,
        missing_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        selected_hosts: List[Dict[str, Any]] = []
        if mask is not None:
            for idx in self._iter_bits(mask):
                candidate = self.snapshot.candidates[idx]
                doc = self._candidate_docs[idx]
                metadata = doc["metadata"]
                selected_hosts.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "host": f"{metadata['namespace']}/{metadata['name']}",
                        "cost": candidate.cost,
                        "must_keep": candidate.must_keep,
                    }
                )

        artifact: Dict[str, Any] = {
            "schema": "plan_host_basis/v1",
            "status": status,
            "input_digest": self._digest(),
            "strict_selectors_disabled": self.strict_selectors_disabled,
            "selected_hosts": selected_hosts,
            "obligations": self._witnesses_for(mask),
            "cost": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "total": upper_bound
                if status == STATUS_MINIMUM
                else (self._mask_cost(mask) if mask is not None else None),
            },
            "host_count": len(selected_hosts),
            "bounds_proven_equal": bool(
                mask is not None
                and lower_bound is not None
                and upper_bound is not None
                and lower_bound == upper_bound
            ),
            "counters": {
                "candidates": len(self.snapshot.candidates),
                "obligations": len(self.snapshot.obligations),
                "required_obligations": len(self._required),
                "forbidden_obligations": len(self._forbidden),
                "evaluations": self.evaluations,
                "evaluation_budget": self.snapshot.max_evaluations,
                "budget_exhausted": self.exhausted,
                "nodes_visited": self.nodes_visited,
                "pruned": self.pruned,
                "inactive_candidates": list(self._inactive),
            },
            "incomplete": {
                "missing_secrets": sorted(set(missing_secrets or []))[
                    :MAX_SAMPLE_ENTRIES
                ],
                "missing_files": sorted(set(missing_files or []))[:MAX_SAMPLE_ENTRIES],
            },
            "oracle_failures": self.failures[:MAX_SAMPLE_ENTRIES],
            "warnings": self.warnings[:MAX_SAMPLE_ENTRIES],
            "scope": _SCOPE_NOTE,
        }
        return _cap_artifact_size(artifact)


class _Incomplete(Exception):
    def __init__(self, missing_secrets: List[str], missing_files: List[str]) -> None:
        self.missing_secrets = missing_secrets
        self.missing_files = missing_files


def _join_manifests(parts: Sequence[str]) -> str:
    """
    Flatten manifest YAML parts into one clean multi-document text.

    Candidate manifests may already carry document markers, so naive string
    joining can emit null documents that the real fetcher rejects. Parsing and
    re-serializing the document set keeps the resources identical (the input
    digest is still computed from the caller's original text) while making the
    compiler input unambiguous.
    """
    docs: List[Any] = []
    for part in parts:
        for doc in yaml.safe_load_all(part):
            if doc is not None:
                docs.append(doc)
    return yaml.safe_dump_all(docs)


def _split_ref(obligation_id: str, ref: str, kind: str) -> Tuple[str, str]:
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"obligation {obligation_id} {kind} ref must be 'namespace/name'"
        )
    return parts[0], parts[1]


def _cap_artifact_size(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the serialized artifact within MAX_ARTIFACT_BYTES by trimming only
    optional diagnostic samples; the mandated planning fields are retained.
    """
    optional_lists = ("oracle_failures", "warnings")
    while len(json.dumps(artifact, default=str)) > MAX_ARTIFACT_BYTES:
        trimmed = False
        for key in optional_lists:
            if artifact.get(key):
                artifact[key] = artifact[key][:-1]
                trimmed = True
        incomplete = artifact.get("incomplete") or {}
        for key in ("missing_secrets", "missing_files"):
            if incomplete.get(key):
                incomplete[key] = incomplete[key][:-1]
                trimmed = True
        if not trimmed:
            break
    return artifact


def plan_host_basis(
    snapshot: HostBasisSnapshot,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Plan the minimum-cost Host subset satisfying the snapshot's obligations.

    The return value is a planning artifact (see module docstring); no
    Kubernetes resource is written or deleted and no caller input is mutated.
    """
    active_logger = logger or logging.getLogger("ambassador.plan_host_basis")
    planner = _HostBasisPlanner(snapshot, active_logger)
    planner._validate()

    early = planner._build_model()
    if early is not None:
        return early

    try:
        return planner.search()
    except _Incomplete as exc:
        return planner._incomplete(exc.missing_secrets, exc.missing_files, None)
