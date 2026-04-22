"""
GMSH .msh mesh reader (formats 2.2 and 4.1).

Parses the mesh into numpy arrays suitable for FEM assembly.
No external dependencies beyond numpy.

Outputs
-------
MeshData dataclass:
  nodes     : (N, 2)  float64  node coordinates [m]
  tri       : (E, 3)  int32    triangular element node indices
  tri_tags  : (E,)    int32    physical surface tag per element
  edge_nodes: (B, 2)  int32    boundary line node pairs
  edge_tags : (B,)    int32    physical line tag per boundary edge
  tag_map   : dict    physical tag -> list of element indices
  btag_map  : dict    physical tag -> list of boundary edge indices

Usage
-----
    from Bohemien_Motor_Designer.fea.mesh_reader import read_msh
    mesh = read_msh("PMSM.msh")
    stator_elements = mesh.tag_map[1]   # physical surface 1 = stator iron
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class MeshData:
    nodes:      np.ndarray          # (N,2) float64
    tri:        np.ndarray          # (E,3) int32  -- linear triangles
    tri_tags:   np.ndarray          # (E,)  int32
    edge_nodes: np.ndarray          # (B,2) int32
    edge_tags:  np.ndarray          # (B,)  int32
    tag_map:    Dict[int, List[int]] = field(default_factory=dict)
    btag_map:   Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        # Build tag_map
        self.tag_map = {}
        for i, t in enumerate(self.tri_tags):
            self.tag_map.setdefault(int(t), []).append(i)
        # Build btag_map
        self.btag_map = {}
        for i, t in enumerate(self.edge_tags):
            self.btag_map.setdefault(int(t), []).append(i)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_elements(self):
        return len(self.tri)

    def summary(self) -> str:
        lines = [
            f"MeshData: {self.n_nodes} nodes  {self.n_elements} triangles",
            f"  Physical surfaces: {sorted(self.tag_map.keys())}",
            f"  Physical lines:    {sorted(self.btag_map.keys())}",
        ]
        for tag, elems in sorted(self.tag_map.items()):
            lines.append(f"    PS {tag:4d}: {len(elems):6d} elements")
        return "\n".join(lines)


def read_msh(path: str) -> MeshData:
    """
    Read a GMSH .msh file (format 2.2 or 4.1).
    Returns a MeshData with linear triangles only.
    Second-order nodes are stripped to their corner nodes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    text = p.read_text(encoding="utf-8", errors="replace")
    fmt  = _detect_format(text)

    if fmt >= 4.0:
        return _read_msh4(text)
    else:
        return _read_msh2(text)


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(text: str) -> float:
    for line in text.splitlines():
        if line.strip().startswith("$MeshFormat"):
            continue
        parts = line.strip().split()
        if parts and parts[0].replace(".", "").isdigit():
            try:
                return float(parts[0])
            except ValueError:
                pass
    return 2.2


# ── MSH 2.2 parser ────────────────────────────────────────────────────────────

def _read_msh2(text: str) -> MeshData:
    """Parse GMSH format 2.2"""
    lines = text.splitlines()
    idx   = 0

    # ── Nodes ──
    while idx < len(lines) and "$Nodes" not in lines[idx]:
        idx += 1
    idx += 1  # skip $Nodes header
    n_nodes = int(lines[idx].strip()); idx += 1
    nodes = np.zeros((n_nodes, 2), dtype=np.float64)
    node_id_map = {}   # GMSH 1-based ID -> 0-based index

    for ni in range(n_nodes):
        parts = lines[idx + ni].strip().split()
        gmsh_id = int(parts[0]) - 1
        node_id_map[int(parts[0])] = ni
        nodes[ni, 0] = float(parts[1])
        nodes[ni, 1] = float(parts[2])
    idx += n_nodes

    # ── Elements ──
    while idx < len(lines) and "$Elements" not in lines[idx]:
        idx += 1
    idx += 1
    n_elems = int(lines[idx].strip()); idx += 1

    tri_list       = []
    tri_tags_list  = []
    edge_list      = []
    edge_tags_list = []

    for ei in range(n_elems):
        parts = list(map(int, lines[idx + ei].strip().split()))
        elem_type  = parts[1]
        n_tags     = parts[2]
        phys_tag   = parts[3] if n_tags >= 1 else 0
        node_start = 3 + n_tags

        if elem_type == 2:   # 3-node triangle
            ns = [node_id_map[parts[node_start + k]] for k in range(3)]
            tri_list.append(ns)
            tri_tags_list.append(phys_tag)
        elif elem_type == 9:  # 6-node triangle (2nd order) -- use corner nodes only
            ns = [node_id_map[parts[node_start + k]] for k in range(3)]
            tri_list.append(ns)
            tri_tags_list.append(phys_tag)
        elif elem_type == 1:  # 2-node line
            ns = [node_id_map[parts[node_start + k]] for k in range(2)]
            edge_list.append(ns)
            edge_tags_list.append(phys_tag)
        elif elem_type == 8:  # 3-node line (2nd order) -- use end nodes
            ns = [node_id_map[parts[node_start + k]] for k in range(2)]
            edge_list.append(ns)
            edge_tags_list.append(phys_tag)

    tri       = np.array(tri_list,       dtype=np.int32) if tri_list       else np.zeros((0,3), np.int32)
    tri_tags  = np.array(tri_tags_list,  dtype=np.int32) if tri_tags_list  else np.zeros(0, np.int32)
    edge_nodes = np.array(edge_list,     dtype=np.int32) if edge_list      else np.zeros((0,2), np.int32)
    edge_tags  = np.array(edge_tags_list,dtype=np.int32) if edge_tags_list else np.zeros(0, np.int32)

    return MeshData(nodes=nodes, tri=tri, tri_tags=tri_tags,
                    edge_nodes=edge_nodes, edge_tags=edge_tags)


# ── MSH 4.1 parser ────────────────────────────────────────────────────────────

def _read_msh4(text: str) -> MeshData:
    """Parse GMSH format 4.1"""
    lines = text.splitlines()
    idx   = 0

    def skip_to(section):
        nonlocal idx
        while idx < len(lines) and section not in lines[idx]:
            idx += 1
        idx += 1

    # ── Nodes ──
    skip_to("$Nodes")
    header = lines[idx].strip().split(); idx += 1
    n_blocks   = int(header[0])
    total_nodes = int(header[1])

    nodes = np.zeros((total_nodes, 2), dtype=np.float64)
    node_id_map = {}
    global_ni = 0

    for _ in range(n_blocks):
        bh = lines[idx].strip().split(); idx += 1
        n_in_block = int(bh[3])
        node_ids = []
        for k in range(n_in_block):
            node_ids.append(int(lines[idx + k].strip()))
        idx += n_in_block
        for k in range(n_in_block):
            parts = lines[idx + k].strip().split()
            node_id_map[node_ids[k]] = global_ni
            nodes[global_ni, 0] = float(parts[0])
            nodes[global_ni, 1] = float(parts[1])
            global_ni += 1
        idx += n_in_block

    nodes = nodes[:global_ni]

    # ── Elements ──
    skip_to("$Elements")
    header = lines[idx].strip().split(); idx += 1
    n_blocks = int(header[0])

    tri_list       = []
    tri_tags_list  = []
    edge_list      = []
    edge_tags_list = []

    for _ in range(n_blocks):
        bh = lines[idx].strip().split(); idx += 1
        # entity_dim, entity_tag, elem_type, n_elems_in_block
        entity_dim = int(bh[0])
        entity_tag = int(bh[1])
        elem_type  = int(bh[2])
        n_in_block = int(bh[3])

        for k in range(n_in_block):
            parts = list(map(int, lines[idx + k].strip().split()))
            # parts[0] = element id, parts[1:] = node ids
            node_ids = parts[1:]
            if elem_type in (2, 9):   # tri3 or tri6
                ns = [node_id_map[node_ids[j]] for j in range(3)]
                tri_list.append(ns)
                tri_tags_list.append(entity_tag)
            elif elem_type in (1, 8): # line2 or line3
                ns = [node_id_map[node_ids[j]] for j in range(2)]
                edge_list.append(ns)
                edge_tags_list.append(entity_tag)
        idx += n_in_block

    tri        = np.array(tri_list,        dtype=np.int32) if tri_list       else np.zeros((0,3), np.int32)
    tri_tags   = np.array(tri_tags_list,   dtype=np.int32) if tri_tags_list  else np.zeros(0, np.int32)
    edge_nodes = np.array(edge_list,       dtype=np.int32) if edge_list      else np.zeros((0,2), np.int32)
    edge_tags  = np.array(edge_tags_list,  dtype=np.int32) if edge_tags_list else np.zeros(0, np.int32)

    return MeshData(nodes=nodes, tri=tri, tri_tags=tri_tags,
                    edge_nodes=edge_nodes, edge_tags=edge_tags)
