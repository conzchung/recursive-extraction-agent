from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from datetime import datetime, timedelta

from models import init_llm, GPT54_args

from typing import Union


async def check_documents(
    candidates: dict,
    rules: list[str],
    selected_llm_args: dict = GPT54_args,
    ) -> dict:
    """Send extraction candidates + rules to the LLM for compliance scoring.

    Each rule is evaluated independently and concurrently (up to 20 at a
    time).  The LLM returns a JSON object per rule with ``score`` (0-10),
    ``reason``, and ``evidence_markdown``.

    Args:
        candidates: Mapping of extraction_id to extraction result string.
        rules: List of natural-language validation instructions (already
            brace-escaped for PromptTemplate).
        selected_llm_args: LLM config dict; defaults to GPT-5.4.

    Returns:
        A tuple of ``(result_list, token_usage)`` where *result_list* is
        a list of score/reason/evidence dicts (one per rule) and
        *token_usage* contains prompt/completion/total token counts.
    """
    template = """
You are an expert compliance reviewer. Given multiple extraction results (JSON-like objects) from different sources, produce an assessment with:
1) a numeric score (0–10),
2) a concise reason,
3) human-verifiable evidence in Markdown.

INPUTS
- candidates: a JSON object mapping extraction_id -> extraction_result
  - extraction_id is the display name of the source (use it verbatim in evidence tables)
  - extraction_result is a JSON string (already json.dumps). Parse it into an object before reasoning.
- rule: a natural language instruction describing what to verify
  - rule may request comparisons, validations, calculations, anomaly checks, completeness checks, etc.

OUTPUT (STRICT)
Return ONLY valid JSON (RFC 8259) with exactly these keys:
- "score": integer 0..10
- "reason": string
- "evidence_markdown": string (Markdown)

Do not output any other keys. Do not output any text outside the JSON object.

GENERAL PRINCIPLES
- Evidence must be sufficient for a user to quickly verify the "reason" and "score".
- Every concrete value mentioned in "reason" MUST appear in "evidence_markdown".
- Only include evidence relevant to the rule (avoid dumping entire documents).
- If a required value is missing in a source, write "Not Found" (never blank).
- Use semantic field matching when names differ but meaning is clear in context:
  - Company name -> company_name/account_name/name/beneficiary_name
  - Account holder -> account_name/acct_name/acc_holder
  - Address -> registered_address/address/residential_address
  - Date -> issue_date/transaction_date/date
  - Late Penality -> Late Penalty (spelling variant)
- Normalize strings for matching: trim, lowercase, collapse multiple spaces; ignore trivial punctuation differences when appropriate.
- If calculations are required by the rule:
  - Show inputs and computed outputs inside evidence_markdown.
  - If any required input is missing, state what is missing and do not claim equality.
- Keep source order identical to the provided candidates order.
- Lease duration comparison (when rule involves leasing period/duration):
  - Treat the lease end date as INCLUSIVE. Compute duration using end_exclusive = end_date + 1 day, then derive the calendar duration between start_date and end_exclusive.
  - If sources share the same start_date and end_date but duration text differs, treat it as a PASS if the inclusive-derived duration supports equivalence; record it as a representation difference in evidence.
- **Output language**: Detect the language of `rule`. Produce `reason`, Markdown headings
  (e.g. "## Evidence", "### Violations", "### Sample matches", "### Matches", "### Notes / mappings / assumptions"),
  and table column labels (e.g. "Case/Topic", "Issue", "Why it matches") in that same language.
  Everything else stays in English: JSON keys, the literal "Not Found" token, and the Coverage line
  (total_cases, pass, fail, missing).

SCORING GUIDANCE (0–10)
- 10: All required values found across all relevant sources and all match exactly after normalization; no missing values.
- 9: All found values match, but one or more required values are "Not Found".
- 7–8: Minor differences (short forms, punctuation, small formatting) but clearly same meaning.
- 4–6: Partial match or mixed results; some pass, some fail.
- 1–3: Major mismatch with weak connection.
- 0: Completely inconsistent/unrelated OR rule cannot be satisfied at all from evidence.

NEVER give score=10 if any required value used in the decision is "Not Found".

EVIDENCE MARKDOWN REQUIREMENTS
The evidence_markdown MUST follow this structure (headings and table labels in the same language as the rule):

## Evidence
**Sources:** <extraction_id_1>, <extraction_id_2>, ...
**Rule focus:** <short restatement of what was checked>
**Coverage:** total_cases=<N>, pass=<P>, fail=<F>, missing=<M>
**Display strategy:** show violations only if pass_rate >= 0.80; otherwise show matches only (pass_rate = pass/total_cases)

Then include ONE of the following sections depending on pass_rate:

A) If pass_rate >= 0.80, include:
### Violations (only)
A Markdown table with ALL sources as columns:
| ID | Case/Topic | <extraction_id_1> | <extraction_id_2> | ... | Issue |
|---:|---|---|---|---|---|

- Include only failing or missing cases.
- Each row is one "case" as defined by the rule (often per line item / per period / per month).
- "Issue" should be short: "Not equal", "Missing in <source>", "Out of range", "Inconsistent inputs", etc.

If there are ZERO violations, still include:
### Sample matches (at least 5)
Show at least 5 representative passing cases (or all if fewer than 5 exist), using table:
| ID | Case/Topic | <extraction_id_1> | <extraction_id_2> | ... | Why it matches |
|---:|---|---|---|---|---|

B) If pass_rate < 0.80, include:
### Matches (only)
Show only passing cases (to avoid overwhelming users when many cases fail), table with ALL sources as columns:
| ID | Case/Topic | <extraction_id_1> | <extraction_id_2> | ... | Why it matches |
|---:|---|---|---|---|---|

Optionally (if helpful for verification), add:
### Notes / mappings / assumptions
- Bullet list describing semantic mappings used, normalization used, calculation assumptions, and any important omissions.

CASE DEFINITION
- Derive "cases" from the rule:
  - If rule is per month/per period/per fee item: treat each month/period/item as one case (row).
  - If rule is a single scalar validation: one case.
  - If rule explicitly says otherwise, follow it.

CONSISTENCY RULES
- Use extraction_id values exactly as provided for table column headers.
- Do not include JSONPath or internal paths unless explicitly requested; show only the source name and extracted value.
- Keep numeric formatting as in source when possible; do not invent currency conversions.

INPUTS
candidates:
{candidates}

rule:
{rule}

Now:
1) Parse each candidates[extraction_id] JSON string into an object.
2) Apply the rule.
3) Produce ONLY the JSON output with keys: score, reason, evidence_markdown.
4) REMINDER: reason, Markdown headings, and table labels must be in the same language as the rule.
"""

    prompt = PromptTemplate(template=template, input_variables=['candidates', 'rule'])
    
    selected_llm_args['reasoning_effort'] = 'medium'
    llm = init_llm(selected_llm_args)
    chain = prompt | llm | JsonOutputParser()

    params = []
    for rule in rules:
        param = {
            'candidates': candidates,
            'rule': rule
            }
        params.append(param)

    with get_openai_callback() as cb:
        result_list = await chain.abatch(params, config={"max_concurrency": 20})

    token_usage = {
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
    }
    
    return result_list, token_usage