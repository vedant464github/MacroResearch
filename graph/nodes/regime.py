"""
Reads QuantRisk's exported regime state (data/regime/regime_export.json).
This is the integration point between the two portfolio projects - deliberately
kept as a simple file read rather than importing QuantRisk's notebook code
directly, since notebook code isn't meant to be imported and doing so would
create a fragile cross-project dependency.

Staleness check: if the export is more than STALE_THRESHOLD_DAYS old, flag it
rather than silently presenting old regime data as current - a risk tool that
can't tell the user "this read might be stale" is worse than one that says
nothing.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from graph.state import GraphState, RegimeContext

REGIME_EXPORT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "regime" / "regime_export.json"
STALE_THRESHOLD_DAYS = 7


def regime_node(state: GraphState) -> GraphState:
    if not state.get("needs_regime"):
        return {}

    errors = list(state.get("errors", []))

    if not REGIME_EXPORT_PATH.exists():
        errors.append(f"regime_node: no export found at {REGIME_EXPORT_PATH} - run the QuantRisk export cell")
        return {"regime_context": None, "errors": errors}

    try:
        raw = json.loads(REGIME_EXPORT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"regime_node: malformed regime_export.json: {e}")
        return {"regime_context": None, "errors": errors}

    exported_at = raw.get("exported_at")
    stale = True
    if exported_at:
        try:
            exported_dt = datetime.fromisoformat(exported_at)
            age_days = (datetime.now(timezone.utc) - exported_dt).days
            stale = age_days > STALE_THRESHOLD_DAYS
        except ValueError:
            pass

    regime_context: RegimeContext = {
        "as_of_date": raw.get("as_of_date", "unknown"),
        "current_regime": raw.get("current_regime", "unknown"),
        "regime_probability": raw.get("regime_probability", 0.0),
        "prior_regime": raw.get("prior_regime", "unknown"),
        "regime_since": raw.get("regime_since", "unknown"),
        "key_indicators": raw.get("key_indicators", {}),
        "stale": stale,
    }

    if stale:
        errors.append(f"regime_node: regime export is stale (exported {exported_at}) - re-export from QuantRisk")

    return {"regime_context": regime_context, "errors": errors}
