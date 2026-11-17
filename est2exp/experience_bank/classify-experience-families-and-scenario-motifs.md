# Parallel Training Deployment Experience Scenario Classification Skill

> **Purpose**: Use slow-GPU distribution scenarios, deployment heuristics expressed as formulas, and scoring or simulation validation results to identify reusable families of deployment experience. Then analyze the typical slow-GPU distribution patterns associated with each family. The workflow applies to topologies with 128 GPUs, 256 GPUs, or more and does not assume a fixed GPU count.

## 1. Core Terminology

| Term | Meaning |
|---|---|
| Experience Family | A higher-level category that groups similar heuristics for parallel training deployment. |
| Family Characteristics | The shared parallel strategy tendencies of a family, expressed as 2–3 key, actionable deployment rules. |
| Family-level Scenario Profile | Overall characteristics shared by multiple scenario motifs within the same family in the slow-GPU distribution space. |
| Scenario Motif | A typical slow-GPU distribution pattern identified within an experience family. |
| Slow-GPU Distribution Features | Normalized features describing slow-GPU density, coverage, local concentration, and balance. |
| Affinity Group | A group of nodes with stronger communication affinity within the group than outside it. |
| Intra-Affinity Imbalance | Differences in slow-GPU proportions among nodes within the same affinity group. |
| Inter-Affinity Imbalance | Differences in slow-GPU proportions among affinity groups. |
| Scenario-specific Deployment Experience | Ratios, ranges, or constraints specific to a scenario motif, added to the family-level framework. |
| Deployment Rationale | An explanation of why a deployment strategy suits a scenario, based on both the slow-GPU distribution and the deployment rules. |
| Representative Scenarios | Typical samples used to explain and review a scenario motif. |

## 2. General Principles

Keep two analytical spaces distinct throughout the workflow:

1. **Deployment experience space**: Form experience families from deployment experience formulas.
2. **Slow-GPU scenario space**: Explain scenario motifs within each experience family using slow-GPU distribution features.

Follow this direction of analysis:

$$
\text{Deployment experience formulas}
\rightarrow
\text{Experience families}
\rightarrow
\text{Scenario motif explanations}
$$

Do not predefine experience families directly from scenario features such as the number of affinity groups covered or the total number of slow GPUs.

In addition:

- Use the best strategies identified by scoring and simulation only for anomaly checks and post hoc validation, not to revise the experience formulas retroactively.
- Prefer ratios, ranges, `min/max`, and relationships between variables; avoid hard-coding absolute values that apply only at a particular scale.
- Derive scenario motif names and thresholds from the current data rather than fixing them in advance.
- The Deployment Rationale explains performance mechanisms to help readers understand the results; it is not a rigorous proof of the scoring formula. Explanations must connect scenario distributions with parallel strategy mechanisms instead of offering only abstract conclusions.

---

## 3. Phase I — Data Preparation

### Step 1 — Input Normalization

**Goal**: Convert reports from different deployment scales and sources into a common data structure.

Fields to normalize include:

- Total GPU count: $N$.
- Number of nodes: $M$.
- GPUs per node: $K$.
- Number of affinity groups: $A$.
- Nodes per affinity group: $R$.
- Slow-GPU count on node $i$: $c_i$.
- Slow-GPU count in affinity group $j$: $s_j$.
- Total slow-GPU count: $S$.
- Deployment experience formulas: split into fields such as `PP-rule`, `TP-rule`, `DP-rule`, `EP-rule`, `MB-rule`, and `active-rule`.
- Strategies with the best score and the best simulated performance, referred to below as the scoring-optimal and simulation-optimal strategies: retain only for validation.

Also verify that the counts are consistent:

$$
S=\sum_{i=1}^{M} c_i
$$

and:

$$
S=\sum_{j=1}^{A} s_j.
$$

### Step 2 — Anomaly Screening and Isolation

**Goal**: Set aside unreliable samples and those requiring review before deriving deployment rules.

Distinguish at least four anomaly categories:

| Anomaly Type | Checks | Follow-up |
|---|---|---|
| Data anomaly | The sum of node-level slow-GPU counts differs from the reported total; affinity-group totals differ from node totals; fields are out of bounds or missing. | Exclude and reintroduce only after correction. |
| Experience formula anomaly | Hard constraints conflict with one another, or the experience formula explicitly excludes the scenario's own scoring-optimal strategy. | Include in the anomaly table; exclude from the analysis used to construct experience families and scenario motifs. |
| Scoring–simulation inconsistency | The scoring-optimal and simulation-optimal strategies differ substantially. | Flag the scoring approach for review; temporarily exclude the sample from the evidence used to derive reliable deployment rules. |
| Suspected experience outlier | Similar scenarios have highly consistent deployment rules, but one scenario has a distinctive rule with no clear explanation. | Mark as Review; do not delete automatically. |

The output must begin with an **anomaly summary** containing the scenario ID, anomaly type, reason, and whether the scenario is excluded.

---

## 4. Phase II — Experience Family Construction

### Step 3 — Experience Feature Selection

**Do not assume in advance that PP, TP, and MB must define the experience families.**

Evaluate PP, TP, DP, EP, MB, the active GPU count, and any other parallelism dimensions individually. Select the features that best distinguish experience families in the current data.

Assess candidate features against four criteria:

| Criterion | Assessment |
|---|---|
| Discriminative power | Does the deployment rule vary substantially across scenarios? A dimension that is identical everywhere is unsuitable as a primary indicator. |
| Stability | Does the rule maintain a stable direction, ratio, or range within a candidate family? |
| Independence | If a dimension is largely derived from others, such as $DP=N/(PP\times TP)$, prefer treating it as an auxiliary rule. |
| Transferability | Prefer `min/max`, ratios, ranges, and variable relationships over absolute constants valid only at the current scale. |

Use a **minimal explanatory set**: start with the most discriminative deployment experience feature, then add features that substantially reduce the grouping of dissimilar rules together. Stop when the main patterns are clearly and consistently separated. Treat the remaining features as auxiliary rules.

### Step 4 — Experience Family Construction

**Goal**: Form experience families using only the main deployment experience features.

Procedure:

- Normalize expressions first. For example, `PP=1`, `PP=min`, and `PP tends to be small` may be grouped under a tendency toward a shallow pipeline, while retaining the original formulas.
- Separate clearly opposing tendencies into different families, such as `PP=1` versus `PP>=4`, or `MB=1` versus `MB>=2PP`.
- Use auxiliary rules for DP, EP, and the active GPU count to check compatibility and refine distinctions within a family.
- **Do not use** scenario features such as slow-GPU density, affinity-group coverage, or local concentration to define experience families at this step.

### Step 5 — Family-level Experience Abstraction

Family-level deployment experience summarizes stable tendencies shared by the member scenarios. It should not be obtained by simply intersecting all of their formulas.

For example, members of the same family may have:

$$
PP\ge4,\qquad TP\le2,\qquad MB\ge2PP
$$

$$
PP\ge4,\qquad TP\le\frac{PP}{2},\qquad MB\ge2PP
$$

and:

$$
PP\ge2,\qquad TP\le\frac{PP}{2},\qquad MB\ge\max(PP,8).
$$

At the family level, these can be abstracted as:

- Pipeline depth tends to be moderate or high.
- TP is small relative to PP.
- MB is large relative to PP and often increases with PP.

In the final main table, **do not create a separate “Family-level Deployment Experience” column**. Incorporate the principal deployment experience derived here directly into Family Characteristics:

1. Begin with a concise structural profile, such as “deep pipeline” or “shallow pipeline.”
2. List 2–3 key, actionable recommendations or relationships for PP/TP/MB, or whichever parameters are most prominent in the current data.
3. Keep shared hard constraints, derived relationships, and auxiliary rules involving DP, EP, the active GPU count, and similar dimensions in the body text; do not repeat them in the main table.

Retain specific upper and lower bounds in the scenario-specific deployment experience.

---

## 5. Phase III — Scenario Motif Discovery

### Step 6 — Scenario Feature Construction

Use **dimensionless ratios** wherever possible so the features remain reusable for 128-GPU, 256-GPU, and larger systems.

#### 6.1 Global Slow-GPU Ratio

Definitions:

- $S$: Total slow-GPU count in the current scenario.
- $N$: Total GPU count in the cluster.

$$
\rho=\frac{S}{N}
$$

Physical meaning: The proportion of slow GPUs across the entire cluster, indicating the overall severity of the slow-GPU problem.

#### 6.2 Node Coverage

Definitions:

- $c_i$: Slow-GPU count on node $i$.
- $M$: Total number of nodes.
- $\#\{i:c_i>0\}$: Number of nodes containing at least one slow GPU.

$$
C_N=\frac{\#\{i:c_i>0\}}{M}
$$

Physical meaning: The proportion of nodes that contain at least one slow GPU.

#### 6.3 Affinity-group Coverage

Definitions:

- $s_j$: Slow-GPU count in affinity group $j$.
- $A$: Total number of affinity groups.
- $\#\{j:s_j>0\}$: Number of affinity groups containing at least one slow GPU.

$$
C_A=\frac{\#\{j:s_j>0\}}{A}
$$

Physical meaning: The proportion of affinity groups that contain at least one slow GPU.

#### 6.4 Maximum Per-node Slow-GPU Ratio

Definitions:

- $c_i$: Slow-GPU count on node $i$.
- $K$: Number of GPUs per node.

$$
R_{\max}=\max_i\frac{c_i}{K}
$$

Physical meaning: **The highest fraction of slow GPUs on any single node.** It indicates whether slow GPUs are highly concentrated on a single node. For example, $R_{\max}=1$ means that at least one node consists entirely of slow GPUs.

#### 6.5 Intra-Affinity Manhattan Imbalance

First define the slow-GPU proportion of node $p$ in affinity group $j$:

- $c_{j,p}$: Slow-GPU count on node $p$ in affinity group $j$.
- $K$: GPUs per node.
- $R$: Nodes per affinity group.

$$
r_{j,p}=\frac{c_{j,p}}{K}
$$

For affinity group $j$, define the mean pairwise difference in node slow-GPU proportions:

$$
D_{\mathrm{intra},j}
=\frac{2}{R(R-1)}
\sum_{p<q}|r_{j,p}-r_{j,q}|.
$$

For the entire scenario, use the maximum imbalance across affinity groups:

$$
D_{\mathrm{intra}}=\max_jD_{\mathrm{intra},j}.
$$

When each affinity group contains exactly two nodes:

$$
D_{\mathrm{intra},j}=|r_{j,1}-r_{j,2}|.
$$

Physical meaning: Whether the slow-GPU burden is balanced among nodes within the same affinity group. A larger value indicates greater local imbalance in some affinity group, such as one node having a high proportion of slow GPUs while another has few or none.

#### 6.6 Inter-Affinity Manhattan Imbalance

First define the affinity-group slow-GPU proportion:

- $s_j$: Slow-GPU count in affinity group $j$.
- $R\times K$: Total GPU count in one affinity group.
- $A$: Total number of affinity groups.

$$
a_j=\frac{s_j}{RK}.
$$

Then define the mean pairwise difference in slow-GPU proportions across affinity groups:

$$
D_{\mathrm{aff}}
=\frac{2}{A(A-1)}
\sum_{p<q}|a_p-a_q|.
$$

Physical meaning: Whether the slow-GPU burden is balanced across affinity groups. Larger values indicate greater concentration in a few affinity groups; smaller values indicate a more even distribution across groups.

> Entropy may be retained as an auxiliary feature. However, if its definition cannot distinguish zero slow GPUs from one slow GPU, it cannot be the sole basis for scenario classification.

### Step 7 — Scenario Feature Selection

This process mirrors Step 3: **do not assume in advance that $C_A$ or $D_{\mathrm{intra}}$ must define the scenario motifs.**

Within each experience family, compare:

- $\rho$: Global slow-GPU ratio.
- $C_N$: Node coverage.
- $C_A$: Affinity-group coverage.
- $R_{\max}$: Maximum per-node slow-GPU ratio.
- $D_{\mathrm{intra}}$: Intra-affinity imbalance.
- $D_{\mathrm{aff}}$: Inter-affinity imbalance.

Prioritize **comparisons near family boundaries**: scenarios with similar global slow-GPU ratios and coverage but different experience families are especially informative.

After comparison, further divide scenario features into two levels:

- **Primary distinguishing features**: Structural features that determine how the motif differs from neighboring motifs. Give these priority in the final main table.
- **Auxiliary distinguishing features**: Features that describe severity, boundaries, or the sample range, but do not define the motif's core identity.

For example, structural features such as affinity-group coverage and intra-affinity imbalance are often strong candidates for primary distinguishing features. Continuous quantities such as the global slow-GPU ratio and maximum per-node slow-GPU ratio are more suitable as auxiliary features if they merely fluctuate among samples. **However, primary and auxiliary roles must not be fixed in advance**: if a continuous quantity actually determines a scenario boundary, promote it to a primary distinguishing feature.

A shallow decision tree or rule list may help identify candidate features. These rules are explanatory only and must not redefine the experience families retroactively.

### Step 8 — Scenario Motif Extraction and Scenario-specific Experience

Within each experience family, use the prominent scenario features selected in Step 7 to form motifs such as A1, A2, B1, and B2.

Each scenario motif must include:

- Scenario motif name.
- Scenario count.
- Slow-GPU distribution features, listing **primary distinguishing features** first, followed by necessary **auxiliary distinguishing features**.
- Representative scenarios.
- Scenario-specific deployment experience.
- Deployment rationale.

Express scenario-specific experience as:

$$
E_{\mathrm{scenario}}
=
E_{\mathrm{family}}
+
\Delta E_{\mathrm{scenario}}.
$$

Start with the shared family rules, then add the ratios, ranges, or upper and lower bounds specific to the motif.

If a motif contains only one sample, label it a “boundary/special scenario motif” to avoid overgeneralization.

For each deployment rule, explain how it relates to the current motif. Use **performance explanations accessible to readers unfamiliar with parallel training**. Follow the sequence “scenario characteristics → main performance issue → why a parallel parameter is adjusted → resulting performance benefits/costs,” instead of giving only abstract judgments such as “reduced local isolation benefits” or “high synchronization cost.”

Consider the following mechanisms, and include only those relevant to the current scenario and deployment rules:

- **PP (pipeline parallelism)**: Increasing PP divides the model into more pipeline stages, which can reduce the model size and memory requirements per stage and may confine the impact of a locally slow region to fewer stages. However, more stages generally increase pipeline bubbles, inter-stage transfers, and waiting between stages. Reducing PP decreases these pipeline dependencies and suits scenarios where slow GPUs are already widespread and additional stages offer little local isolation.
- **TP (tensor parallelism)**: Increasing TP distributes computation within a layer across more GPUs, reducing per-GPU computation and memory pressure. However, GPUs within the same TP group synchronize more frequently, so one slow GPU may make the entire synchronization group wait. Smaller TP is therefore often preferred when local slow-GPU concentration is pronounced. When the pipeline is shallow and more parallelism within the model is needed to share computation, moderate or larger TP may be appropriate, while avoiding excessively large synchronization groups.
- **DP (data parallelism)**: Increasing DP adds parallel replicas, potentially improving overall throughput and using GPUs freed by reducing PP or TP. The cost is gradient synchronization between replicas. Explain DP in terms of communication costs and whether it is primarily derived from constraints such as `N/(PP×TP)`.
- **MB (number of microbatches)**: Larger MB can fill a deep pipeline more fully and reduce the proportion of pipeline bubbles. When the pipeline is shallow or PP=1, fewer microbatches are needed to fill it, or there is no pipeline to fill, so MB can be smaller. Larger MB may also increase activation memory requirements; explain this in conjunction with the original experience constraints.
- **EP (expert parallelism, if present)**: Explain whether it distributes expert computation/capacity, and the expert communication and load-balancing costs of increasing EP. If EP is only a shared range or hard constraint in the current experience, do not present it as a main explanation for the family's behavior.

If using a phrase such as “reduced local isolation benefits,” immediately explain its concrete meaning. For example: “Once slow GPUs are distributed across multiple affinity groups or many nodes, further increasing PP makes it difficult to confine them to a few pipeline stages. The bubbles and waiting introduced by additional stages may therefore outweigh the isolation benefits.”

### Step 9 — Family-level Scenario Abstraction

**Perform this step only after the scenario motifs have been established.**

Review all scenario motifs within an experience family and summarize the main characteristics of their slow-GPU distributions rather than computing a strict intersection.

The family-level scenario profile should retain only **the main scenario features that characterize the family**. Do not include every computable metric. Keep auxiliary statistics in the motif-level Slow-GPU Distribution Features column.

Use an **“abstract summary + supporting data for the main features”** format, for example:

> **Broad coverage with a generally balanced distribution** (many scenarios cover all affinity groups; when only some groups are covered, the slow-GPU proportion difference between the two nodes within an affinity group is usually small).

Or:

> **Localized distribution with more pronounced intra-affinity imbalance** (slow GPUs mainly cover only a few affinity groups; in scenarios covering two affinity groups, the within-group difference in slow-GPU proportions between the two nodes, `D_intra`, is often large).

When using variable names in the overall profile, state the physical meaning before the variable, for example:

- `Coverage of 2/4 affinity groups: C_A = 0.50`.
- `Large within-group difference in slow-GPU proportions between the two nodes: D_intra ≥ 0.50`.

Data in parentheses or in the supporting part of a sentence should support only the **main features**. Do not list every auxiliary statistic, such as the global slow-GPU ratio, node coverage, and maximum per-node slow-GPU ratio.

Use language such as “usually,” “mainly,” “predominantly,” and “often when…”. Use categorical statements such as “all motifs satisfy this condition” only when they hold for every motif.

---

## 6. Final Main Table Specification

Before the main table, add a brief explanation titled **“Intuitive Effects of Parallel Parameters on Performance.”** Its purpose is to help readers without a parallel training background understand the Deployment Rationale column, rather than to derive the deployment rules again. Focus on the four core parameters PP, TP, DP, and MB; add EP only if it actually distinguishes cases in the current data.

Standard Markdown tables do not support merged cells or precise column widths, so use an **HTML table embedded in Markdown** for the final report's main table.

Requirements:

- When one family spans multiple scenario motifs, merge the Experience Family, Family Characteristics, and Family-level Scenario Profile cells using `rowspan`.
- Use wider columns for Slow-GPU Distribution Features, Scenario-specific Deployment Experience, and Deployment Rationale.
- Use narrower columns for Experience Family and Scenario Count.
- Separate rules within each cell using `;<br>` to make rule-by-rule comparison easier.
- **Do not use LaTeX delimiters inside the HTML table**, such as `$...$` or `$$...$$`, because some Markdown renderers do not parse LaTeX within HTML tables. Use directly displayable mathematical text such as `PP ≥ 4`, `TP/DP ≤ 1/4`, `DP = N/(PP × TP)`, and `ρ = 0`. HTML subscripts are acceptable, such as `C<sub>A</sub> = 0.50` and `D<sub>intra</sub> ≥ 0.50`. Continue using `$$...$$` for standalone formulas in the body text.
- For slow-GPU distribution features, state the physical meaning before the variable name and value. For example: `Coverage of 2/4 affinity groups: C<sub>A</sub> = 0.50;`. Within each cell, list primary distinguishing features before necessary auxiliary distinguishing features.
- Consistently refer to `R_{\max}` as “maximum per-node slow-GPU ratio,” rather than simply “hotspot.”
- Explain performance mechanisms directly without prefixes such as “Explanatory synthesis:”. State once before the table that this column aids understanding and is not a rigorous proof of the scoring formula.
- Prioritize explaining why the pipeline should be deeper or shallower; why TP should be small, moderate, or large; why MB should be larger or smaller; and why DP should increase or be treated primarily as a value derived from other constraints. State the corresponding performance benefits and costs explicitly.

Recommended column order:

1. Experience Family.
2. Family Characteristics (including principal deployment experience).
3. Family-level Scenario Profile.
4. Scenario Motif.
5. Scenario Count.
6. Slow-GPU Distribution Features.
7. Scenario-specific Deployment Experience.
8. Representative Scenarios.
9. Deployment Rationale.

---

## 7. What Question Does Each Main Table Column Answer?

| Column | Main Responsibility / Question Answered |
|---|---|
| Experience Family | Which deployment experiences belong to the same higher-level category? This is the primary classification label, derived from the deployment experience space. |
| Family Characteristics (including principal deployment experience) | What type of parallel structure is this, and which main rules should guide deployment first? Start with a qualitative profile such as “deep pipeline” or “shallow pipeline,” then list 2–3 key, actionable parameter recommendations or ratio relationships. |
| Family-level Scenario Profile | What **main characteristics** do scenarios using this deployment experience exhibit in their slow-GPU distributions overall? Begin with an abstract summary and support it only with data for the main features. |
| Scenario Motif | What typical slow-GPU distribution patterns can be distinguished within this family? Give only the motif name; do not include the count in the name. |
| Scenario Count | How many valid scenarios belong to each motif? This indicates the strength of its sample support. |
| Slow-GPU Distribution Features | Which prominent features identify this motif? List primary distinguishing features first, followed by necessary auxiliary distinguishing features. |
| Scenario-specific Deployment Experience | What specific ratios, upper and lower bounds, or additional rules does this motif require beyond the family's principal rules? |
| Representative Scenarios | Which original scenarios are most useful for explaining and reviewing this motif? Preserve traceability. |
| Deployment Rationale | Why does this motif correspond to this deployment experience? For readers unfamiliar with parallel training, explain PP/TP/DP/MB settings, and EP if needed, through performance mechanisms: how does each parameter affect computation, synchronization, pipeline overhead, or throughput, and why does that effect suit the slow-GPU distribution? |

---

## 8. Principles for Reuse Across Scales

When applying the workflow to 256 or more GPUs, do not copy the number of families, thresholds, or scenario motif names from the 128-GPU report. Reuse only the workflow:

1. Repeat anomaly screening.
2. Reselect the most prominent deployment experience features.
3. Reconstruct experience families.
4. Recalculate scenario features using the new $N,M,K,A,R$.
5. Reselect the most prominent scenario features.
6. Reconstruct scenario motifs.
7. Derive new family-level scenario profiles.
8. Regenerate the nine-column main table with merged cells, scenario counts, and deployment rationales.
