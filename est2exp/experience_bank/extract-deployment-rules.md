---
name: deployment-experience-extractor
description: Extract reusable distributed-training deployment experience from scenario reports by deriving rules from scoring functions, feature definitions, and feasibility constraints. Never infer deployment experience from simulation winners.
---

# Deployment Experience Extractor

## Purpose

Use this skill to extract reusable deployment experience from distributed-training scenario reports.

The goal is **not** to repeat the winning simulated configuration. The goal is to derive the deployment rules that are actually encoded by the scenario's scoring function, feature definitions, and feasibility constraints.

Primary deployment dimensions:

- `active_gpu`
- `PP` — pipeline parallel degree
- `TP` — tensor parallel degree
- `DP` — data parallel degree
- `MB` — microbatch count
- `EP` — expert parallel degree

Prefer reusable symbolic rules over instance-specific numbers.

Examples of useful experience forms:

- minimum / maximum
- fixed value
- preferred interval
- preferred ratio
- threshold / boundary
- monotonic preference
- jointly determined / coupled relation
- topology-aware condition
- memory-aware condition
- cannot be summarized independently

---

# Core Principle

## Derive experience from the score, never from the simulation winner

Authoritative evidence, in descending order:

1. scoring function / cost function
2. feature definitions used by that function
3. feasibility constraints
4. scenario constants needed to eliminate derived features

Simulation outputs such as:

- simulated Top-1
- Top-K
- recommended configuration
- regret
- simulated latency
- simulation rank
- whether score Top-1 matches simulation Top-1

are **validation evidence only**.

Never use simulation results to:

- invent a deployment rule
- choose between score-equivalent candidates
- revise a score-derived rule
- explain why a parameter should take a value
- infer a topology relationship
- resolve an ambiguity that the score itself does not resolve

A simulation winner may be mentioned only after the rule has already been derived independently from the score.

---

# Strict Evidence Boundary

Only use variables and relationships that appear in:

- the score / cost function
- feature definitions
- feasibility constraints
- scenario constants required to evaluate those expressions

Do not add general distributed-training knowledge unless the user explicitly asks for external interpretation.

If the report does not define a feature well enough to derive a parameter rule, state:

> Cannot be determined from the current information.

If a scenario has simulation results but no scenario-specific score formula, do **not** derive scenario-specific deployment experience. You may summarize only common feasibility rules.

---

# Workflow

## Step 1 — Read the scenario configuration and identify constants

Extract only constants that are needed for derivation, for example:

- total active GPUs `N`
- global batch `B`
- total experts `E`
- expert capacity per EP rank `C`
- model layer count
- legal candidate grids for `PP`, `TP`, `EP`, `MB`
- topology information only when topology-derived features are actually used by the score

Typical structural relation:

$$
PP \cdot TP \cdot DP = N
$$

therefore:

$$
DP = \frac{N}{PP \cdot TP}
$$

If all candidates use all GPUs, record:

$$
\text{active\_gpu} = N
$$

as a feasibility fact, not a score preference.

---

## Step 2 — Extract hard feasibility constraints

Separate hard feasibility from scoring preference before analyzing the score.

Examples:

$$
PP \cdot TP \cdot DP = N
$$

$$
EP \mid DP
$$

$$
E/EP \le C
$$

$$
DP \cdot MB \mid B
$$

$$
\mathrm{memory} \le \text{memory\_limit}
$$

If divisibility implies an inequality, the divisibility rule remains the stronger hard constraint.

For example:

$$
DP \cdot MB \mid B
$$

implies:

$$
DP \cdot MB \le B
$$

but do not replace the divisibility rule with the weaker inequality when exact feasibility matters.

When useful, eliminate `DP`:

$$
DP = \frac{N}{PP \cdot TP}
$$

so:

$$
DP \cdot MB \le B
$$

becomes:

$$
\frac{N}{PP \cdot TP} \cdot MB \le B
$$

and therefore:

$$
\boxed{\frac{MB}{PP} \le \frac{B}{N}TP}
$$

For `N=256`, `B=128`:

$$
\boxed{\frac{MB}{PP} \le \frac{TP}{2}}
$$

Label this as **feasibility**, not score preference.

---

## Step 3 — List every score term affecting each deployment parameter

Do not analyze score clauses in isolation.

For each of:

- `PP`
- `TP`
- `DP`
- `MB`
- `EP`
- `active_gpu`

find **all** score terms that directly or indirectly depend on that parameter.

Examples:

- `TP <= 4`
- `TP / DP >= 0.05`
- `MB / PP >= 1`
- `pipeline_bubble <= 0.25`
- `microbatch_pressure <= 0.4`
- `memory_pressure >= 0.30`
- `TP_straggler_exposure <= 0.2`

Then combine all relevant terms before summarizing the parameter.

Do not say `TP <= 4` is the deployment experience if another score term simultaneously prefers `TP >= 4`. The combined result may instead be:

$$
TP = 4
$$

Similarly, do not summarize `MB >= PP` without checking the batch-feasibility upper bound.

---

# Derived Feature Elimination

## Step 4 — Convert derived features back into deployment parameters

The preferred reasoning chain is:

$$
\text{Deployment parameter}
\rightarrow
\text{Derived feature}
\rightarrow
\text{Score}
\rightarrow
\text{Deployment experience}
$$

But the final experience should eliminate intermediate features whenever possible.

Preferred result:

$$
PP,\ TP,\ DP,\ MB,\ EP
$$

rather than an opaque feature threshold.

---

## Common elimination patterns

### Data parallel degree

$$
DP = \frac{N}{PP \cdot TP}
$$

Example:

$$
DP \ge D_0
$$

becomes:

$$
\boxed{PP \cdot TP \le \frac{N}{D_0}}
$$

---

### TP/DP ratio

$$
\frac{TP}{DP}
=
\frac{PP \cdot TP^2}{N}
$$

Therefore:

$$
\frac{TP}{DP} \le r
$$

becomes:

$$
\boxed{PP \cdot TP^2 \le rN}
$$

and:

$$
\frac{TP}{DP} \ge r
$$

becomes:

$$
\boxed{PP \cdot TP^2 \ge rN}
$$

Example:

$$
TP/DP < 0.5
$$

gives:

$$
\boxed{PP \cdot TP^2 < N/2}
$$

---

### MB/PP ratio

$$
MB/PP \ge r
$$

becomes:

$$
\boxed{MB \ge rPP}
$$

Example:

$$
MB/PP \ge 1
$$

becomes:

$$
\boxed{MB \ge PP}
$$

---

### Expert capacity pressure

If:

$$
p_E = \frac{E/EP}{C}
=\frac{E}{C \cdot EP}
$$

then:

$$
p_E \ge q
$$

becomes:

$$
EP \le \frac{E}{qC}
$$

If expert feasibility also requires:

$$
p_E \le 1
$$

then:

$$
EP \ge E/C
$$

and the combined range is:

$$
\boxed{
E/C \le EP \le E/(qC)
}
$$

If the score is monotonically better as `EP` becomes smaller, summarize the experience as:

> `EP = min feasible`

Do **not** automatically write `EP = 1`.

---

### Pipeline bubble

If:

$$
\mathrm{bubble} = \frac{PP-1}{MB+PP-1}
$$

then a lower-bound score condition:

$$
\mathrm{bubble} \ge q
$$

gives:

$$
PP-1 \ge q(MB+PP-1)
$$

and therefore:

$$
\boxed{
MB \le \frac{1-q}{q}(PP-1)
}
$$

A score may intentionally reward a **higher** bubble value. Preserve the literal score direction even when it is counterintuitive.

For an upper bound:

$$
\mathrm{bubble} \le q
$$

derive the corresponding lower bound on `MB`.

---

### Microbatch pressure

If:

$$
p_M=
\frac{1}{1+B/(DP \cdot MB)}
=
\frac{DP \cdot MB}{DP \cdot MB+B}
$$

then:

$$
p_M \le q
$$

becomes:

$$
\boxed{
DP \cdot MB \le \frac{q}{1-q}B
}
$$

and:

$$
p_M \ge q
$$

becomes:

$$
\boxed{
DP \cdot MB \ge \frac{q}{1-q}B
}
$$

Substitute:

$$
DP=\frac{N}{PP\cdot TP}
$$

when useful.

---

### Microbatch size

If:

$$
\text{microbatch\_size}=
\frac{B}{DP \cdot MB}
$$

then:

$$
\text{microbatch\_size} \ge s
$$

becomes:

$$
\boxed{
DP \cdot MB \le B/s
}
$$

---

# Score Interpretation Rules

## Step 5 — Treat score conditions as soft preferences unless they are feasibility constraints

A condition in the score is not automatically a deployment constraint.

Distinguish:

**Hard feasibility**

$$
PP \cdot TP \cdot DP=N
$$

from:

**Soft score preference**

$$
TP\le4 \Rightarrow +20
$$

Never convert a soft score threshold into a hard deployment constraint without qualification.

Use wording such as:

- strongly prefers
- full-reward region
- avoids penalty
- secondary preference
- weak tie-break
- can be sacrificed when a higher-weight term dominates

---

## Step 6 — Combine weights, not just thresholds

When score terms conflict, compare their actual contribution.

Do not mechanically collect all thresholds and pretend they must all hold simultaneously.

Example:

- `DP >= 64` gives `+20`
- a ratio term improves gradually when `TP/DP` increases

If leaving `DP >= 64` gains only a few points from the ratio term but loses `+20`, the experience is:

> Prefer the high-DP region first, then maximize `TP/DP` within that region.

Do not write the unattainable ratio platform as a hard target.

---

## Step 7 — Analyze piecewise functions exactly

Respect:

- strict `<` vs `<=`
- discontinuities
- plateaus
- hinge slopes
- step bonuses
- exact boundary behavior

A strict threshold can make the best region lie just below a boundary.

Example:

- bonus if `r < 0.25`
- bonus disappears at exactly `r = 0.25`

Then the experience may be:

> Keep `r < 0.25` and approach `0.25` from below.

Do not simplify this to `r <= 0.25`.

---

## Step 8 — Keep tiny nonzero terms

Do not delete tiny terms such as:

$$
10^{-6}
$$

If they affect ranking, preserve them as:

> weak tie-break

Do not promote them to a primary deployment rule.

---

# Coupled Parameters

## Step 9 — Do not freeze other variables at the simulated winner

If a parameter is coupled with others, analyze the symbolic relation.

Bad:

> With the Top-1 configuration fixed at `PP=4`, TP should be 2.

Good:

$$
TP \le PP/2
$$

If score + feasibility jointly determine:

$$
PP=4,\quad TP=2,\quad MB=4
$$

show the derivation.

If they do **not** determine a unique value, say:

> Cannot be summarized independently.

or:

> `PP`, `TP`, and `MB` are jointly determined by the following relations.

Never use the simulation winner to resolve remaining score ties.

---

# Topology Rules

## Step 10 — Retain topology features when they cannot be eliminated

Examples:

- `TP_scope_exposure`
- `DP_scope_exposure`
- `TP_straggler_exposure`
- `DP_straggler_exposure`
- slow-link exposure

If the report defines the topology calculation sufficiently, you may evaluate it for candidate parameter values.

However:

- do not infer topology semantics from numeric coincidence
- do not claim `DP = affinity_group_count` merely because the numbers match
- do not claim `TP = GPUs_per_node` merely because the numbers match

Only make such a statement if the score or feature definition causally uses that topology structure.

If a topology feature cannot be reduced to PP/TP/DP/MB/EP, preserve it as an irreducible condition:

$$
\boxed{\text{DP\_scope\_exposure} \le 0.25}
$$

Do not erase it merely because the current discrete grid collapses to one configuration.

---

# Memory Rules

## Step 11 — Treat memory score terms literally

Memory may appear as:

- feasibility upper bound
- score upper threshold
- score lower threshold

These are different.

Examples:

Hard feasibility:

$$
\mathrm{memory} \le 57.6\,\mathrm{GiB}
$$

Score preference:

$$
\text{memory\_pressure} \ge 0.30
$$

The latter may intentionally reward **higher** memory pressure. Preserve the score's literal direction.

If memory depends on multiple variables and cannot be algebraically reduced cleanly, keep:

$$
\text{memory\_pressure} \ge q
$$

as an irreducible coupled condition.

You may evaluate the report's own memory formula over the legal parameter grid, but do not substitute a simulation result for that calculation.

---

# Current-Grid Collapse

## Step 12 — Derive symbolic experience first, then optionally collapse to the current grid

Always present the reusable symbolic rule before the scenario-specific discrete result.

Example:

Symbolic experience:

$$
TP \le PP/2,\qquad
1.3PP \le MB \le 2.7PP
$$

Current grid may then collapse to:

$$
PP=4,\ TP=2,\ MB=8
$$

Label this explicitly as:

> current-grid collapse

The current-grid collapse is **not** the same thing as the general deployment experience.

---

# Per-Parameter Summary

After completing all derivations, summarize the six deployment dimensions.

Recommended forms:

### active_gpu
- fixed by feasibility
- maximize / minimize
- cannot determine

### PP
- `PP = min`
- `PP = max`
- `PP >= k`
- `PP <= k`
- coupled with TP/MB
- cannot independently summarize

### TP
- fixed
- upper/lower bound
- ratio to PP
- jointly determined
- topology-aware

### DP
Prefer deriving from:

$$
DP=N/(PP\cdot TP)
$$

rather than treating DP as independent when it is not.

### MB
Remember:

> `MB` means **microbatch count**, not per-microbatch batch size.

### EP
Use:

- fixed feasible minimum
- preferred range
- monotonic preference
- cannot independently summarize

Do not silently equate `EP=min` with `EP=1`.

---

# Validation Status

Scenario status such as:

- `PASS_DEBUG`
- `DEGRADED`
- `FAIL_QUALITY`

may be retained as a **confidence / validation label**.

It must not change the score-derived rule.

Recommended interpretation:

- `PASS_DEBUG`: score passed the scenario-local debug acceptance criteria
- `DEGRADED`: formula may still be extracted, but validation confidence is lower
- `FAIL_QUALITY`: formula can still be described, but save it with low confidence

Do not treat any of these as independent cross-scenario validation unless the report explicitly says so.

---

# Multi-Scenario Rule

When analyzing multiple scenarios:

1. summarize every scenario separately
2. do not merge scenarios merely because they have the same current-grid winner
3. preserve the different causal rules that produced the same configuration
4. only create a cross-scenario pattern after all individual derivations are complete

For example, several scenarios may collapse to:

`PP1_TP4_DP64_EP8_MB1`

for completely different reasons:

- high-DP preference
- memory threshold
- microbatch pressure
- pipeline bubble
- TP/DP boundary
- topology exposure

Those are different deployment experiences.

---

# Required Output Format

For each scenario, use this structure:

## Scenario `<id>`

### 1. Score-derived preferences

List only terms that materially affect deployment.

### 2. Algebraic elimination

Show the important transformations, for example:

$$
DP=\frac{N}{PP\cdot TP}
$$

$$
TP/DP<0.5
\Rightarrow
PP\cdot TP^2<N/2
$$

### 3. Combined deployment rule

Combine all relevant score terms and feasibility constraints.

Clearly distinguish:

- hard feasibility
- primary score preference
- secondary score preference
- weak tie-break
- irreducible topology / memory condition

### 4. Parameter summary

State deployment experience for:

- `active_gpu`
- `PP`
- `TP`
- `DP`
- `MB`
- `EP`

Use `cannot independently summarize` when appropriate.

### 5. Current-grid collapse

Only if the score + feature definitions + feasibility constraints uniquely or meaningfully narrow the current candidate grid.

Never use simulation Top-1 to create this collapse.

### 6. Confidence

Record report status separately from the rule itself.

### 7. One-line experience

End with a compact reusable statement.

---

# Preferred Style

- Derive before summarizing.
- Prefer symbolic rules before substituting scenario numbers.
- Prefer direct PP/TP/DP/MB/EP relations over repeating feature names.
- Eliminate intermediate features whenever possible.
- Preserve irreducible topology and memory conditions when necessary.
- Explicitly state tradeoffs when score terms conflict.
- Do not hide strict inequalities or discontinuities.
- Do not infer causal topology rules from coincidental numeric matches.
- Do not overfit the experience to the current candidate grid.
- Do not use simulation winners as evidence for deployment rules.

---

# Self-Check Before Finalizing

Verify all of the following:

- [ ] Every deployment claim is supported by score, feature definition, or feasibility constraint.
- [ ] No rule was inferred from simulation Top-1 or Top-K.
- [ ] Hard constraints are separated from score preferences.
- [ ] All score terms affecting each parameter were combined.
- [ ] Derived features were algebraically eliminated where possible.
- [ ] Strict inequalities and piecewise discontinuities were preserved.
- [ ] Tiny nonzero terms were retained as tie-breaks.
- [ ] Coupled parameters were not summarized independently without justification.
- [ ] Topology semantics were not inferred from numeric coincidence.
- [ ] `MB` was treated as microbatch count.
- [ ] Symbolic experience was presented before current-grid collapse.
- [ ] Validation status was kept separate from the deployment rule.
- [ ] If the score is missing or definitions are insufficient, the answer says so instead of guessing.
