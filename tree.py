from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Iterable, Tuple, List
import math
@dataclass
class Node:
    """
    Scenario-tree node storing belief and propagated states.
    """
    id: int
    depth: int
    parent: Optional[int]
    p: float  # branch probability (node weight w_n)
    opp_u: Optional[float] = None
    tag: Optional[str] = None
    belief: Dict[str, float] = field(default_factory=dict)
    x_e: Optional[tuple] = None
    x_o: Optional[tuple] = None
@dataclass
class Tree:
    root: int
    nodes: dict[int, Node] = field(default_factory=dict)
    def add(self, node: Node):
        self.nodes[node.id] = node
    def children(self, nid: int):
        return [n for n in self.nodes.values() if n.parent == nid]
    def traverse(self):
        stack = [self.root]
        while stack:
            nid = stack.pop(0)
            yield self.nodes[nid]
            stack.extend([c.id for c in self.children(nid)][::-1])
