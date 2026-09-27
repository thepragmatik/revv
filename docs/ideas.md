# Hypothesis ledger: fast local typed decisions

**[Home](../README.md) · [Plain-English view](start-here.md) · [Novelty audit](novelty-audit.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md) · [Next steps](next-steps.md)**

The original prompt asks us to invent and rigorously challenge combinations of ModernBERT, CLIP-style learning and System-1 decision models. Combining familiar parts may produce a real contribution, but it must have a reason, a derivation and a test that could prove it wrong.

## Current verdict

A broad hypothesis—reuse one state across typed fields and let each question exit at its own layer—is too weak. Nimble already has shared-context multi-field scoring and minimal fact-edit pairs; AdaMTL has task-aware compute policies over a shared encoder and merges task demands; cascaded QA rankers share partial encodings. Global allocation, deployment-tuned exits and token-level routing add still more overlap. See the [overlap audit](novelty-audit.md) and [source ledger](sources.md).

The candidate remains a **hypothesis**, and the closest prior art is now sharper. Incremental Transformer reuses cached representations for appended text and questions; Contiguity, Not Importance directly studies edit-aware cache repair and reports that local unconditional repair can beat importance-based selection when edits are answer-relevant; NormWorlds-CF and CRPO already supervise answer changes and invariance; No Universal Signal compares confidence against update-regression signals; SNR establishes learned routes through shared subnetworks; COAR estimates counterfactual effects of model components for editing; Delta Networks propagates sparse activation changes; and TAPIR learns when to revise outputs during prefix processing. This closes broad claims about cache reuse, task routing, component attribution, delta propagation and adaptive revision. The current preregistered screen asks only whether typed-output-level refresh selection after an input edit adds value at a fixed verifier budget. It is not a novelty claim and may be a small systems variation. See the [fifth overlap pass](sources.md#fifth-overlap-pass-counterfactual-component-modeling-and-shared-routing) and [sixth overlap pass](sources.md#sixth-overlap-pass-sparse-delta-propagation-and-adaptive-revision).

A crucial distinction: the screen learned signal is probability of a label change, not expected benefit from running the verifier. It also measures whether routed refreshes improve answer quality over cached answers and whether end-to-end reference CPU time falls. A change-recall win without a quality gain and measured savings is a no-go. Its uniform edit sampler includes many no-effect edits, so a passing result still needs a relevant-edit and independent-data challenge. See the [primary-source overlap entries](sources.md#fourth-overlap-pass-edit-aware-inference-and-answer-changes) and [preregistration](../experiments/kaggle/2026-09-27-intervention-risk-probe.md).

## Candidate architecture weave to screen

1. **Shared state cache:** test a small ModernBERT-like or compact recurrent encoder that processes the state once; benchmark against tiny CPU models. Keep token/sentence features if the task depends on evidence location.
2. **Cheap typed match:** represent question+option or field criteria and candidate evidence in a shared space. Use CLIP-style contrastive learning only when exact support spans and hard negatives exist and the bi-encoder retrieval floor is adequate.
3. **Selective verifier:** a shallow scorer returns typed probabilities first; uncertain fields can pay for question-conditioned cross-attention over a small evidence set or a deeper field adapter. Include a no-escalation and abstain path.
4. **Intervention masks:** create minimal fact edits with verified labels. Train affected fields to follow their edited gold labels and regularize unaffected field distributions to remain stable.
5. **Risk-priced allocation:** estimate each stage's conditional gold-task loss and measured CPU cost. Escalate if expected loss reduction beats its cost, or solve a bundle budget allocation when the runtime couples fields.
6. **Bundle-level evaluation:** report any-error risk, exact-match bundle accuracy, calibration and p95 latency as question count grows.

Every item has close prior art. The possible contribution lies in sparse intervention supervision shaping which typed field receives costly verification, with end-to-end CPU gains—not in a component name. The first probe must establish that per-field verifier work is large enough to save.

## Mathematical objective

For request features $x$, question depths $d_q$, shared cost $A_x$, branch costs $B_q(x,\cdot)$, per-request estimated task losses $r_q(x,\cdot)$, and latency weight $\lambda$:

$J_x(\mathbf d)=\sum_q r_q(x,d_q)+\lambda\left[A_x(\max_q d_q)+\sum_q B_q(x,d_q)\right].$

For additive expected loss, fixing maximum depth makes per-question choices separable, with one question forced to reach that maximum. The exact $O(QL)$ solver and assumptions are in the [novelty audit](novelty-audit.md) and [technical derivation](architecture.md). Bundle-any-error constraints are not separable and need separate held-out risk evaluation.

## Cost-sensitive repair value (derived target, not yet tested)

For a cached answer $a_j^0$, a verifier answer $a_j^1$, post-edit gold label $y_j'$, and task loss $\ell$, the useful target is the conditional value of refreshing:

$V_j=\mathbb{E}[\ell(a_j^0,y_j')-\ell(a_j^1,y_j')\mid s,e,q_j].$

With additive per-field cost $c_j$ and a Lagrange price $\lambda$, refresh field $j$ only when $V_j>\lambda c_j$. Under a fixed cardinality budget, sorting by expected loss reduction is optimal by an exchange argument; with a shared batch/setup cost, the cost is coupled and this separability no longer holds. A change-probability head $P(y_j'\ne y_j)$ is only a proxy: it can rank a field high even when the verifier is likely to make a correct cached answer wrong, or low when the cache is already wrong but a verifier could fix it. This motivates a later repair-value head only if the current frozen screen shows a real quality/savings gap. It is a derivation and test proposal, not a novelty or empirical claim.

## Follow-on candidate: local rebase, then verifier-value routing

This is a **separate, untested architecture hypothesis**. It does not change the frozen screen above.

1. When a fact changes, first refresh the edited sentence/span representation and any explicitly dependent local features. Do not ask a learned router whether to leave an answer-relevant edit wholly unrepaired. If dependencies are not bounded, widen the re-encoding region or fall back to full recomputation.
2. Recompute cheap typed outputs from the refreshed local state. For each field, estimate verifier value as expected reduction in gold-task loss, $V_j=\mathbb E[\ell(p_j^{local},y'_j)-\ell(p_j^{verify},y'_j)\mid x_j]$.
3. Spend the expensive verifier budget on fields with the largest positive value relative to measured cost. Compare against unconditional local repair, confidence routing, random routing, and full recomputation. When retrieval or batching creates shared costs, choose the subset using measured joint cost rather than independent thresholds.
4. Treat a low-rank edit-to-logit correction as an optional ablation, not a presumed contribution. Test exactness and calibration after repeated edits; Delta Networks show that thresholded delta approximations can accumulate drift. Keep a periodic exact rebase or a calibrated full-recompute fallback.

This weave combines local representation repair with cost-sensitive decision-theoretic verification. Incremental encoders, stale-cache repair, adaptive revision, shared task routing and expected-value allocation are all established ingredients. The possible remaining claim is only their interaction for typed outputs after arbitrary state edits, and it survives only if an independent workload shows lower gold loss and lower end-to-end CPU cost at the same budget. The current Q20 synthetic screen does not test this full design.

## Secondary quality weave: intervention-local bundle learning

For a valid state edit $e$ with known affected field set $A(e)$, train on both typed answer vectors and penalize distribution changes on unaffected fields:

$$L_{pair}=\sum_q CE(p_q(s),y_q)+\sum_q CE(p_q(s^e),y_q^e)+\beta\sum_{q\notin A(e)}JS(p_q(s),p_q(s^e)).$$

This combines shared state encoding, typed heads and contrastive/counterfactual evidence supervision. Nimble and PairCFR cover counterfactual data; TACL/logic-guided work covers QA contrast and consistency; AdaMTL covers task-aware compute sharing; QA cascades cover selective refinement. Only the use of a verified sparse affected-field mask to train stage-specific task-risk routing across one typed bundle remains a possible distinction. First measure targeted flips, unaffected-field spillover, retrieval recall and branch cost; do not train if these do not show headroom.

## Rejected, demoted and open ideas

| Idea | Disposition | Reason |
| --- | --- | --- |
| Generic shared state, question heads or independent early exit | Baselines | Strong prior art includes common-context multi-question generation and adaptive QA resource allocation. |
| Basic counterfactual evidence pairs as the novelty | Demote | Nimble directly publishes minimal fact edits that change typed labels. A bundle-wide affected-field mask is a narrower secondary hypothesis, not a new claim yet. |
| Compiled criteria plus evidence matching as the novelty | Demote | RULERS and other rubric/evidence systems overlap; our ContractNLI evidence-use gates did not pass. |
| CLIP-style MaxSim or candidate-set interaction | Test only when justified | Late interaction and option-set scoring exist; untrained MaxSim lost to pooled scoring in our pilot. |
| Query-aware token pruning | Separate future branch | TR-BERT, CoLT5 and document-QA token-pruning work are close. Include only if profiling shows token encoding is the bottleneck. |
| Shared-cost-aware joint depth choice by itself | Demote to control | AdaMTL directly covers task-aware compute over a shared encoder; use the solver to compare policies, not as a novelty claim. |
| Intervention-supervised selective verification over typed fields | **Screen first** | Potentially useful weave; test exact target flips, unaffected spillover, evidence recall and local branch cost before any training. |
| CLIP-style evidence matching | Conditional component | Use only if exact evidence labels show a useful retrieval ceiling; our earlier ContractNLI evidence pipeline did not pass. |

## Falsifiers

- The task-risk gated verifier offers no latency/risk improvement over a fixed cascade, tuned margin threshold or compact direct scorer.
- Most CPU time is in the shared encoder, while each field branch is too cheap to route; one hard field keeps shared work alive.
- Exit and compaction overhead erase saved work on CPU.
- A compact existing typed model already passes quality, probability, latency and memory gates.
- Gains appear only against the backbone's own errors, not task labels, or disappear on an independent source/composition split.

Continue prior-art search and small probes; do not start training to validate novelty.
