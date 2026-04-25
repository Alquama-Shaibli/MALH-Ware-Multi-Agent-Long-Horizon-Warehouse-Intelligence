"""
warehouse_env/negotiation.py
============================
Theme #1 — Explicit Negotiation + Coalition Formation

Implements a lightweight negotiation protocol on top of the existing
multi-agent environment. Agents can:

  claim   — exclusively reserve an item or the charger
  release — relinquish a previously held claim
  propose — offer a deal to another agent (e.g. "I take item1, you take item2")
  accept  — accept an outstanding proposal (creates an agreement)
  reject  — decline an outstanding proposal

NegotiationEngine is a pure stateless logic class; all mutable state
lives in NegotiationState (stored inside StateManager).
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ── Data containers (plain dicts for JSON serializability) ─────────────────────

def _make_offer(from_agent: str, to_agent: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "offer_id":   str(uuid.uuid4())[:8],
        "from_agent": from_agent,
        "to_agent":   to_agent,
        "payload":    payload,    # e.g. {"i_take": "item1", "you_take": "item2"}
        "status":     "pending",  # pending | accepted | rejected | expired
    }


def _make_agreement(offer: Dict[str, Any], step: int) -> Dict[str, Any]:
    return {
        "agreement_id": str(uuid.uuid4())[:8],
        "agents":       [offer["from_agent"], offer["to_agent"]],
        "payload":      offer["payload"],
        "formed_step":  step,
        "fulfilled":    False,
        "broken":       False,
    }


# ── NegotiationState ───────────────────────────────────────────────────────────

class NegotiationState:
    """
    All mutable negotiation state. One instance lives inside StateManager
    and is reset each episode.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # {resource_key -> agent_id}  resource_key = "charger" or "item:<name>"
        self.claim_map: Dict[str, str] = {}
        # offer_id -> offer dict
        self.outstanding_offers: Dict[str, Dict[str, Any]] = {}
        # list of agreement dicts
        self.active_agreements: List[Dict[str, Any]] = []
        # rolling event log (last 20 events)
        self.event_log: List[Dict[str, Any]] = []
        # stats
        self.proposals_sent: int = 0
        self.agreements_formed: int = 0
        self.agreements_broken: int = 0
        self.spam_count: int = 0   # proposals sent to same target within 3 steps

        self._last_proposal_step: Dict[str, int] = {}  # agent_id -> last proposal step

    def to_info(self) -> Dict[str, Any]:
        return {
            "claim_map":             dict(self.claim_map),
            "outstanding_offers":    list(self.outstanding_offers.values()),
            "active_agreements":     list(self.active_agreements),
            "negotiation_events":    list(self.event_log[-10:]),
            "agreement_success_rate": round(
                self.agreements_formed / max(self.proposals_sent, 1), 3
            ),
            "proposals_sent":        self.proposals_sent,
            "agreements_formed":     self.agreements_formed,
        }


# ── NegotiationEngine ─────────────────────────────────────────────────────────

class NegotiationEngine:
    """
    Pure logic — processes one negotiation action and returns
    (reward_delta, event_description).
    Caller is responsible for applying the reward delta.
    """

    # Rewards / penalties
    CLAIM_BONUS          =  0.02   # successful, uncontested claim
    CLAIM_CONFLICT       = -0.05   # resource already claimed by other
    PROPOSAL_SPAM        = -0.02   # repeat proposal within 5 steps
    AGREEMENT_BONUS      =  0.08   # both agents form an agreement
    REJECT_NEUTRAL       =  0.00
    RELEASE_NEUTRAL      =  0.00
    AGREEMENT_BREAK      = -0.10   # agent acts against its own agreement

    SPAM_WINDOW          = 5       # steps

    def process(
        self,
        agent_id: str,
        action_type: str,
        ns: NegotiationState,
        current_step: int,
        payload: Optional[Dict[str, Any]] = None,
        target_agent: Optional[str] = None,
        offer_id: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Returns (reward_delta, event_description).
        """
        payload = payload or {}

        if action_type == "claim":
            return self._handle_claim(agent_id, ns, current_step, payload)
        elif action_type == "release":
            return self._handle_release(agent_id, ns, current_step, payload)
        elif action_type == "propose":
            return self._handle_propose(agent_id, target_agent, ns, current_step, payload)
        elif action_type == "accept":
            return self._handle_accept(agent_id, offer_id, ns, current_step)
        elif action_type == "reject":
            return self._handle_reject(agent_id, offer_id, ns, current_step)

        return 0.0, ""

    # ── Claim ──────────────────────────────────────────────────────────────────
    def _handle_claim(
        self, agent_id: str, ns: NegotiationState,
        step: int, payload: Dict[str, Any],
    ) -> Tuple[float, str]:
        resource = payload.get("resource", "charger")
        existing = ns.claim_map.get(resource)
        if existing and existing != agent_id:
            ns.event_log.append({"step": step, "type": "claim_conflict",
                                  "agent": agent_id, "resource": resource})
            return self.CLAIM_CONFLICT, f"{agent_id} tried to claim {resource} (held by {existing})"
        ns.claim_map[resource] = agent_id
        ns.event_log.append({"step": step, "type": "claim",
                              "agent": agent_id, "resource": resource})
        return self.CLAIM_BONUS, f"{agent_id} claimed {resource}"

    # ── Release ────────────────────────────────────────────────────────────────
    def _handle_release(
        self, agent_id: str, ns: NegotiationState,
        step: int, payload: Dict[str, Any],
    ) -> Tuple[float, str]:
        resource = payload.get("resource", "charger")
        if ns.claim_map.get(resource) == agent_id:
            del ns.claim_map[resource]
            ns.event_log.append({"step": step, "type": "release",
                                  "agent": agent_id, "resource": resource})
        return self.RELEASE_NEUTRAL, f"{agent_id} released {resource}"

    # ── Propose ────────────────────────────────────────────────────────────────
    def _handle_propose(
        self, agent_id: str, target_agent: Optional[str],
        ns: NegotiationState, step: int, payload: Dict[str, Any],
    ) -> Tuple[float, str]:
        if not target_agent:
            return -0.01, "propose with no target"

        # Spam detection
        last = ns._last_proposal_step.get(agent_id, -999)
        if step - last < self.SPAM_WINDOW:
            ns.spam_count += 1
            ns.event_log.append({"step": step, "type": "proposal_spam", "agent": agent_id})
            return self.PROPOSAL_SPAM, f"{agent_id} spamming proposals"

        offer = _make_offer(agent_id, target_agent, payload)
        ns.outstanding_offers[offer["offer_id"]] = offer
        ns.proposals_sent += 1
        ns._last_proposal_step[agent_id] = step
        ns.event_log.append({"step": step, "type": "propose",
                              "agent": agent_id, "target": target_agent,
                              "offer_id": offer["offer_id"], "payload": payload})
        return 0.0, f"{agent_id} proposed to {target_agent}: {payload}"

    # ── Accept ─────────────────────────────────────────────────────────────────
    def _handle_accept(
        self, agent_id: str, offer_id: Optional[str],
        ns: NegotiationState, step: int,
    ) -> Tuple[float, str]:
        offer = ns.outstanding_offers.get(offer_id or "")
        if not offer or offer["to_agent"] != agent_id:
            return 0.0, f"{agent_id} tried to accept unknown offer {offer_id}"

        offer["status"] = "accepted"
        agreement = _make_agreement(offer, step)
        ns.active_agreements.append(agreement)
        ns.agreements_formed += 1
        del ns.outstanding_offers[offer["offer_id"]]
        ns.event_log.append({"step": step, "type": "agreement_formed",
                              "agents": agreement["agents"],
                              "agreement_id": agreement["agreement_id"],
                              "payload": agreement["payload"]})
        return self.AGREEMENT_BONUS, (
            f"Agreement formed: {agreement['agents'][0]} & {agreement['agents'][1]} "
            f"— {agreement['payload']}"
        )

    # ── Reject ─────────────────────────────────────────────────────────────────
    def _handle_reject(
        self, agent_id: str, offer_id: Optional[str],
        ns: NegotiationState, step: int,
    ) -> Tuple[float, str]:
        offer = ns.outstanding_offers.pop(offer_id or "", None)
        if offer:
            offer["status"] = "rejected"
            ns.event_log.append({"step": step, "type": "rejected",
                                  "agent": agent_id, "offer_id": offer_id})
        return self.REJECT_NEUTRAL, f"{agent_id} rejected {offer_id}"

    # ── Agreement compliance check ──────────────────────────────────────────────
    def check_agreement_violations(
        self, agent_id: str, action_type: str,
        ns: NegotiationState, current_step: int,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """
        Call before every non-negotiation action.
        Returns (penalty, description) if the action violates an agreement.
        """
        payload = payload or {}
        for agr in ns.active_agreements:
            if agr["broken"] or agent_id not in agr["agents"]:
                continue
            p = agr["payload"]
            # If agent agreed to "not take" an item but now picks it
            you_take = p.get("you_take") if agr["agents"][1] == agent_id else p.get("i_take")
            other_take = p.get("i_take") if agr["agents"][1] == agent_id else p.get("you_take")
            if action_type == "pick" and other_take and payload.get("item") == other_take:
                agr["broken"] = True
                ns.agreements_broken += 1
                ns.event_log.append({"step": current_step, "type": "agreement_broken",
                                      "agent": agent_id, "agreement_id": agr["agreement_id"]})
                return self.AGREEMENT_BREAK, f"{agent_id} broke agreement {agr['agreement_id']}"
        return 0.0, ""
