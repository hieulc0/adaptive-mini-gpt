"""Phase 2: Valid Structural Experiment Generation by Construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(frozen=True)
class StructuralProbe:
    """A structurally grounded candidate query generated from environment premises."""
    probe_id: str
    source: str
    target: str
    probe_type: str  # DIRECTION_REVERSAL, TWO_HOP_BRIDGE, THREE_HOP_CHAIN, CUT_VERTEX_PROBE, DISCONNECTED_OPEN_WORLD
    path_length: int
    question_text: str
    structural_features: tuple[float, float, float, float]  # (is_reverse, path_len_norm, src_degree_norm, tgt_degree_norm)


@dataclass
class ProbeGenerationStats:
    """Diagnostic statistics on candidate probe generation."""
    total_probes_generated: int = 0
    probe_type_counts: dict[str, int] = field(default_factory=dict)
    validity_rate: float = 1.00  # 100% by construction
    generation_duration_seconds: float = 0.0


class StructuralProbeGenerator:
    """Generates valid candidate structural probes covering symmetry, transitivity, and boundaries."""

    def __init__(self, entities: list[str], base_edges: list[tuple[str, str]], relation_name: str) -> None:
        self.entities = entities
        self.base_edges = base_edges
        self.relation_name = relation_name
        self.entity_set = set(entities)

        # Build adjacency and degree maps
        self.adj: dict[str, set[str]] = {e: set() for e in entities}
        self.in_degree: dict[str, int] = {e: 0 for e in entities}
        self.out_degree: dict[str, int] = {e: 0 for e in entities}

        for u, v in base_edges:
            if u in self.adj and v in self.entity_set:
                self.adj[u].add(v)
                self.out_degree[u] += 1
                self.in_degree[v] += 1

    def generate_candidate_probes(self, max_probes: int = 24) -> tuple[list[StructuralProbe], ProbeGenerationStats]:
        """Generate candidate probes guaranteed valid, non-reflexive, and non-redundant."""
        t0 = time.perf_counter()
        probes: list[StructuralProbe] = []
        seen_pairs: set[tuple[str, str]] = set()
        type_counts: dict[str, int] = {}

        max_deg = max(max(self.in_degree.values(), default=1), max(self.out_degree.values(), default=1), 1)

        def add_probe(src: str, tgt: str, p_type: str, path_len: int) -> None:
            if src == tgt:
                return
            if (src, tgt) in seen_pairs:
                return
            if src not in self.entity_set or tgt not in self.entity_set:
                return
            seen_pairs.add((src, tgt))

            is_rev = 1.0 if p_type == "DIRECTION_REVERSAL" else 0.0
            feat = (
                is_rev,
                min(1.0, path_len / 4.0),
                min(1.0, (self.in_degree[src] + self.out_degree[src]) / (2.0 * max_deg)),
                min(1.0, (self.in_degree[tgt] + self.out_degree[tgt]) / (2.0 * max_deg)),
            )

            p_id = f"probe_{len(probes)+1:02d}_{p_type[:6].lower()}"
            q_text = f"Does {src} {self.relation_name} {tgt}?"
            probe = StructuralProbe(
                probe_id=p_id,
                source=src,
                target=tgt,
                probe_type=p_type,
                path_length=path_len,
                question_text=q_text,
                structural_features=feat,
            )
            probes.append(probe)
            type_counts[p_type] = type_counts.get(p_type, 0) + 1

        # 1. Directional Reversals: tests symmetry
        for u, v in self.base_edges:
            add_probe(v, u, "DIRECTION_REVERSAL", path_len=1)

        # 2. Two-Hop Bridges: tests transitivity (u -> v -> w)
        for u in self.entities:
            for v in self.adj[u]:
                for w in self.adj[v]:
                    if w != u:
                        add_probe(u, w, "TWO_HOP_BRIDGE", path_len=2)

        # 3. Three-Hop Chains: tests multi-hop reachability (u -> v -> w -> z)
        for u in self.entities:
            for v in self.adj[u]:
                for w in self.adj[v]:
                    for z in self.adj[w]:
                        if z != u and z != v:
                            add_probe(u, z, "THREE_HOP_CHAIN", path_len=3)

        # 4. Cut-Vertex / Bridge-Removal Probes: queries between endpoints across central hubs
        high_degree_nodes = sorted(
            self.entities, key=lambda x: self.in_degree[x] + self.out_degree[x], reverse=True
        )
        if len(high_degree_nodes) >= 2:
            hub = high_degree_nodes[0]
            for leaf in self.entities:
                if leaf != hub and leaf not in self.adj[hub]:
                    add_probe(leaf, hub, "CUT_VERTEX_PROBE", path_len=2)

        # 5. Disconnected / Open-World Uncertainty Probes: pairs with no path in base edges
        for u in self.entities:
            for v in self.entities:
                if u != v and (u, v) not in seen_pairs and v not in self.adj[u]:
                    add_probe(u, v, "DISCONNECTED_OPEN_WORLD", path_len=4)
                    if len(probes) >= max_probes:
                        break
            if len(probes) >= max_probes:
                break

        # Trim to max_probes
        final_probes = probes[:max_probes]
        dt = time.perf_counter() - t0

        stats = ProbeGenerationStats(
            total_probes_generated=len(final_probes),
            probe_type_counts=type_counts,
            validity_rate=1.00,  # 100% valid by construction
            generation_duration_seconds=dt,
        )

        return final_probes, stats
