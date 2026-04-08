from typing import Any, Dict, List

from validation_utils import check_documents

from logging import getLogger

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = getLogger(__name__)


async def run_validation_workflow(
    candidates: Dict[str, Any],
    rule_sets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run LLM-based compliance checks on extraction candidates.

    Each rule set contains an ``alias`` (display name) and a ``rule``
    (natural-language instruction).  All rules are evaluated concurrently
    via ``check_documents`` and the per-rule scores/evidence are
    aggregated into a single response dict.

    Args:
        candidates: Mapping of extraction_id → extraction result string.
        rule_sets: List of dicts, each with ``alias`` and ``rule`` keys.

    Returns:
        Dict with keys ``validation_result`` (per-rule scores and
        evidence), ``token_usage``, and ``error`` (None on success).
    """

    # Input validation
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Candidates must be a non-empty dict.")
    if not isinstance(rule_sets, list) or not rule_sets:
        raise ValueError("Rule sets must be a non-empty list.")

    for i, rs in enumerate(rule_sets, start=1):
        if not isinstance(rs, dict):
            raise ValueError(f"Rule set at index {i} must be a dict.")
        if "rule" not in rs:
            raise ValueError(f"Rule set at index {i} is missing required key: 'rule'.")
        if "alias" not in rs:
            raise ValueError(f"Rule set at index {i} is missing required key: 'alias'.")
        if not isinstance(rs["rule"], str) or not rs["rule"].strip():
            raise ValueError(f"Rule set at index {i} has an invalid 'rule' value.")
        if not isinstance(rs["alias"], str) or not rs["alias"].strip():
            raise ValueError(f"Rule set at index {i} has an invalid 'alias' value.")

    # Initialize default result structure
    response: Dict[str, Any] = {
        "validation_result": None,
        "token_usage": {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
        "error": None,
    }

    try:
        logger.info(f"Starting validation workflow. rules_count={len(rule_sets)}")

        rules = [rule_set["rule"] for rule_set in rule_sets]
        formatted_rules = [rule.replace("{", "{{").replace("}", "}}") for rule in rules]

        check_results, token_usage = await check_documents(
            candidates=candidates,
            rules=formatted_rules,
        )

        final_check_result: Dict[str, Any] = {}
        for idx, (rule_set, check_result) in enumerate(zip(rule_sets, check_results), start=1):
            final_check_result[f"rule_{idx}"] = {
                "alias": rule_set["alias"],
                "rule": rule_set["rule"],
                "result": check_result,
            }

        response.update(
            {
                "validation_result": final_check_result,
                "token_usage": token_usage or response["token_usage"],
            }
        )

        logger.info("Validation workflow completed successfully.")
        return response

    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        logger.error(error_msg)
        response["error"] = error_msg
        return response

    except KeyError as e:
        error_msg = f"Invalid rule set structure (missing key): {str(e)}"
        logger.error(error_msg)
        response["error"] = error_msg
        return response

    except Exception as e:
        error_msg = f"Unexpected error during validation workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        response["error"] = error_msg
        return response