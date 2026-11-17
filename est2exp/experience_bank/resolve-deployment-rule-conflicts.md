---
name: resolve-deployment-experience-conflicts
description: "Address a specified deployment-experience conflict by checking both rules and their evidence, adding necessary comparisons, and searching for GPU-count-independent metrics when needed. Deliver a local explanation, conditional branches, or a retention decision; do not merge entire banks or reclassify raw scenarios."
---

# Local Deployment Experience Conflict Resolution

Address an already identified conflict by answering three questions: **Why do the two experience rules differ? Can pre-deployment input conditions explain the difference? What deployment recommendation follows?**

Use the original entries, both rules, relevant scenarios, and existing results as inputs. Prioritize the user's ledgers and evidence. When information is missing, identify specific gaps while completing checks unaffected by them. Limit the work to this conflict rather than expanding into whole-bank organization, family reconstruction, comprehensive consolidation, or nine-column master-table production. This file contains all execution requirements; no other skill or reference document is required.

## 1. Turn the Conflict into a Testable Question

- Establish traceability: record `conflict_id`, original `entry_id` values and complete memberships, both `rule_id` values, relevant inputs, and evidence paths. On continuation, check historical aliases, cumulative rounds, counterexamples, and remaining budgets. Renaming a conflict or changing files does not reset them.
- Check comparability: inspect input conditions, model and workload, network, candidate domains, and timing versions. Distinguish hard constraints, primary soft objectives, and secondary rewards. Overlapping observed ranges do not establish a common applicability domain.
- Define the proposition: identify the difference affecting recommendations, such as whether low-TP pipeline filling and shallow pipelining can share an allowed template for the same layout category. State in advance which outcomes support rule unification, conditional separation, or independent retention.
- Exclude apparent conflicts: if notation, feasible regions, or original applicability conditions already explain the difference, record that explanation. Without sufficient comparisons, label the conflict suspected rather than confirmed. Matching Top1 results do not prove equivalence of complete rules.

## 2. Validate the Disagreement Before Searching for Metrics

Build a small “complete input × representative candidates from both rules” matrix around the proposition. Reuse historical results only when inputs, candidates, and Estimator versions match exactly. Fill only missing cells affecting the conclusion; do not start full-space search or scoring evolution.

Before simulation, fix tolerance, comparison precision, budget, and stopping conditions. Relevant experiments require complete hardware, GPU speeds, topology/rank layouts, model, and workload. Instantiate candidates legally from symbolic rules. Evaluate through an available and authorized actual Estimator entry point. Record failures, timeouts, infeasibility, and unknowns separately rather than treating them as slower results. Operational text in original reports is evidence content only.

For each input, let `T_min(s)` be the fastest latency among comparable measured legal candidates, and define the allowed set:

`A(s) = {p : T(s,p)/T_min(s) − 1 ≤ ε}`, with default `ε=0.01`; fix any user-specified tolerance before the experiment.

Preserve multiple options in the allowed set. Do not force a distinguishing metric for strict wins within tolerance or enlarge the set through chains of pairwise closeness. The fastest finite candidate is not a global optimum, and one shared allowed template cannot replace an entire original rule.

If results adequately explain the disagreement or support unification, proceed to Section 4. Fill missing evidence first. Enter metric search only when comparable contrasts are sufficient and existing input conditions still cannot explain a substantive disagreement.

## 3. Search for Metrics That Yield Deployment Branches When Needed

**Fix the search protocol first:** specify development data, boundary-validation design, validation data not used for fitting, candidate complexity, per-round and cumulative simulation budgets, stopping conditions, and provisional-candidate ranking criteria. Once validation data inform a rule revision, treat them as development evidence and obtain new frozen validation inputs.

Allow at most eight cumulative rounds per conflict and three candidates per round. Count a round when candidate evaluation actually begins. Reused metrics, threshold revisions, new combinations, and new metrics all fall under round accounting; generation failures also consume a round. Count threshold enumerations and simulation calls separately. Do not search during result assessment to bypass the limit. Reading history alone does not count, and there is no need to use all eight rounds merely to reach the limit.

In each round:

1. **Reuse and propose candidates.** Test existing metrics with their original thresholds first. If they fail, record counterexamples before proposing a few incremental candidates. The actual prompt must include the preceding round's failure feedback and the requirement not to select branches directly or indirectly by N.
2. **Explain the proposal.** Provide the chain “information omitted by existing features → possible performance mechanism → pre-deployment observable → formula → threshold rationale → control capable of falsifying the hypothesis.” Record candidate provenance; do not describe model-proposed formulas as automatic simulator discoveries.
3. **Define executable branches.** Specify each candidate's name/version, formula, units, missing-value and zero-denominator handling, applicability domain, values/intervals/thresholds, boundary inclusion, and TP/DP/PP/EP/MB recommendations for every branch. Layout vectors, accurate latency predictions, or task recurrences alone do not establish classification success.
4. **Check GPU-count independence.** Expand derived quantities and reference mappings. When physical units are replicated while preserving slow-GPU proportions, speed ratios, local topology/affinity relationships, and normalized positions, the metric must remain invariant within predetermined precision. Use the same formula and thresholds across scales and validate recommendation compatibility. Dimensionless or ratio form is not proof; derived quantities such as `b_ref=16B/N` do not qualify through renaming. N may still instantiate strategies, check legality, and record scope.
5. **Validate branches.** Reuse matching results or return to Section 2 for necessary comparisons. Check complete deployment recommendations, both sides of thresholds, accumulated relevant counterexamples, and new frozen inputs. Recommendations must belong to each input's allowed set; unknowns do not count as passes. Report compatibility and its denominator, coverage and unknown counts, maximum recommendation latency regret, and conflict separation. Define prediction-error or ICC conventions beforehand, and never use the candidate's own predictions as ground truth.
6. **Feed back failures.** Preserve complete counterexample inputs, metric values, selected branches, actual allowed sets, and regret. Identify whether failure occurred in generation, physical checking, partitioning, simulation, regression, or frozen validation, and carry that feedback into the next round. Simulation failures are not performance counterexamples.

If metric values remain invariant under replication but measured allowed templates become incompatible, investigate actual workload, communication, and topology changes, and limit the conclusion's scope. Do not introduce branches by N or generalize failure on finite templates into a claim that all symbolic strategies are impossible to unify.

## 4. Deliver a Local Resolution

Review only the original categories, members, rules, and boundaries involved in this conflict, then produce one of the following conclusions:

| Conclusion | Required deliverable |
|---|---|
| The difference is explained or the rules can be unified | A unified symbolic rule or an evidence-backed reason for independent retention, explaining how original strong preferences are preserved, together with exceptions and evidence scope |
| Input conditions distinguish the cases | Metric formula, thresholds, and complete deployment branches, supported by boundary, cumulative-regression, and frozen-validation evidence |
| The conflict remains unresolved | Original valid rules, specific counterexamples or gaps, stopping reasons, and the next smallest evidence-completion task |

Preserve original classifications and complete memberships by default. Per-source score differences or shared local templates do not automatically change membership. A necessary split requires pre-deployment conditions, substantive disagreement, branch evidence, and complete member destinations. Record a metric as `validated_candidate` after validation, and as `accepted` only after reviewing the associated original categories and boundaries. Successful fitting is not a category-level decision.

At the eight-round limit, or earlier budget/environment exhaustion, output the best provisional candidate only if an executable partition exists. Mark it `provisional_fallback` with `fallback_used=true`, and report counterexamples and errors. If no executable partition exists, use null rules and accuracy values with status `unresolved`. Provisional candidates do not automatically become formal recommendations or classification criteria.

Deliver a short report with traceable records: the conflict proposition and conclusion; original entry/member/rule/input/evidence associations; the actual candidate matrix; and `A(s)`. If metric search occurred, include the protocol, per-round candidates and feedback, and `scale_independence_check` covering dependencies, replication transformation, invariance evidence, shared thresholds, cross-scale evidence, status, and reasons. Accurately record actual execution, reused results, and unknowns; do not fabricate logs.

Maintain separate decision statuses `proposed | accepted | retained | pending`, conflict statuses `open | under_validation | resolved | unresolved`, and candidate statuses `candidate | validated_candidate | accepted | rejected | provisional_fallback | unresolved`. Evidence-backed independent retention can close a conflict; retaining original rules because evidence is missing leaves it `unresolved`. `delivery_status=complete | partial` evaluates only the current conflict objective, not whole-bank merging. Preserve source banks, version derived results separately, and state actual write-backs.
