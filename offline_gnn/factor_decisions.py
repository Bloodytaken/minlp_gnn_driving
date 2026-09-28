"""Structured decision groups for GNN training and confidence-based SCIP fixing."""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F


_GAMMA = re.compile(
    r"^(gamma(?:_bg\d+)?_)(front|back|left|right|f|b|l|r)(\[\d+\])$")
_LANE = re.compile(r"^b_(pos|neg)\[(\d+)\]$")
GAMMA_ORDER = ("front", "back", "left", "right")
LANE_ORDER = ("stay", "pos", "neg")
_DIRECTION = {"f": "front", "b": "back", "l": "left", "r": "right",
              "front": "front", "back": "back", "left": "left", "right": "right"}


def decision_groups(names, candidate_mask):
    candidate = np.asarray(candidate_mask, dtype=bool)
    gamma_raw, lane_raw = {}, {}
    for i, name in enumerate(names):
        if not candidate[i]:
            continue
        match = _GAMMA.match(str(name))
        if match:
            direction = _DIRECTION[match.group(2)]
            gamma_raw.setdefault(match.group(1) + match.group(3), {})[direction] = i
            continue
        match = _LANE.match(str(name))
        if match:
            lane_raw.setdefault(int(match.group(2)), {})[match.group(1)] = i
    gamma = {key: tuple(row[d] for d in GAMMA_ORDER)
             for key, row in gamma_raw.items() if all(d in row for d in GAMMA_ORDER)}
    lane = {node: (row["pos"], row["neg"])
            for node, row in lane_raw.items() if "pos" in row and "neg" in row}
    return gamma, lane


def structured_loss(logits: torch.Tensor, labels: torch.Tensor,
                    names, candidate_mask, root: int = 0,
                    root_lane_weight: float = 2.0):
    """Family-balanced CE: 4-way region groups plus 3-way lane actions."""
    gamma, lane = decision_groups(names, candidate_mask)
    losses = []
    if gamma:
        indices = torch.as_tensor(list(gamma.values()), dtype=torch.long,
                                  device=logits.device)
        scores = logits[indices]
        target = labels[indices].argmax(dim=1)
        losses.append(F.cross_entropy(scores, target))
    if lane:
        for nodes, weight in (([n for n in lane if n != root], 1.0),
                              ([root] if root in lane else [], root_lane_weight)):
            if not nodes:
                continue
            indices = torch.as_tensor([lane[n] for n in nodes], dtype=torch.long,
                                      device=logits.device)
            zero = logits.new_zeros((len(indices), 1))
            scores = torch.cat([zero, logits[indices]], dim=1)
            pair_y = labels[indices]
            target = torch.where(pair_y[:, 0] > 0.5, 1,
                                 torch.where(pair_y[:, 1] > 0.5, 2, 0))
            losses.append(weight * F.cross_entropy(scores, target))
    if not losses:
        raise ValueError("sample contains no complete maneuver decision group")
    return sum(losses)


def structured_fixmap(logits: np.ndarray, names, candidate_mask, root: int,
                      gamma_threshold: float, lane_threshold: float,
                      root_lane_threshold: float):
    """Fix complete decisions whose maximum softmax probability meets the threshold."""
    gamma, lane = decision_groups(names, candidate_mask)
    fixmap, decisions = {}, []
    for key, indices in gamma.items():
        prob = torch.softmax(torch.as_tensor(logits[list(indices)]), 0).numpy()
        winner = int(prob.argmax())
        if float(prob[winner]) >= gamma_threshold:
            for j, index in enumerate(indices):
                fixmap[str(names[index])] = int(j == winner)
            decisions.append(("gamma", key, float(prob[winner]), GAMMA_ORDER[winner]))
    for node, (pos, neg) in lane.items():
        scores = torch.as_tensor([0.0, logits[pos], logits[neg]])
        prob = torch.softmax(scores, 0).numpy()
        winner = int(prob.argmax())
        threshold = root_lane_threshold if node == root else lane_threshold
        if float(prob[winner]) >= threshold:
            fixmap[str(names[pos])] = int(winner == 1)
            fixmap[str(names[neg])] = int(winner == 2)
            decisions.append(("lane", node, float(prob[winner]), LANE_ORDER[winner]))
    return fixmap, decisions, len(gamma), len(lane)


def structured_metrics(logits: np.ndarray, labels: np.ndarray, names,
                       candidate_mask, root: int, gamma_threshold: float,
                       lane_threshold: float, root_lane_threshold: float):
    """Decision-level coverage/error counts, not misleading per-bit accuracy."""
    gamma, lane = decision_groups(names, candidate_mask)
    result = {key: {"groups": 0, "fixed": 0, "wrong": 0}
              for key in ("gamma", "lane_future", "lane_root", "all")}

    def record(family, prediction, target, confidence, threshold):
        row = result[family]
        row["groups"] += 1
        result["all"]["groups"] += 1
        if confidence >= threshold:
            row["fixed"] += 1
            result["all"]["fixed"] += 1
            wrong = int(prediction != target)
            row["wrong"] += wrong
            result["all"]["wrong"] += wrong

    for indices in gamma.values():
        probability = torch.softmax(torch.as_tensor(logits[list(indices)]), 0).numpy()
        record("gamma", int(probability.argmax()),
               int(np.asarray(labels)[list(indices)].argmax()),
               float(probability.max()), gamma_threshold)
    for node, (pos, neg) in lane.items():
        probability = torch.softmax(torch.as_tensor([0., logits[pos], logits[neg]]), 0).numpy()
        pair = np.asarray(labels)[[pos, neg]]
        target = 1 if pair[0] > .5 else (2 if pair[1] > .5 else 0)
        family = "lane_root" if node == root else "lane_future"
        threshold = root_lane_threshold if node == root else lane_threshold
        record(family, int(probability.argmax()), target,
               float(probability.max()), threshold)
    return result
