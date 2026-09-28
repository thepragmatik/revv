# Continue the revv mission

**[Home](../README.md) · [Plain-English idea](../docs/start-here.md) · [Technical design](../docs/architecture.md)**

This folder is the starting point for a new Codex session. It preserves the user's original research brief, the explicit goal of inventing and testing novel architectures, the present facts, working rules and a prompt you can paste into Codex. It is updated as experiments produce results.

```mermaid
flowchart TD
    C["Codex reads this handoff"] --> P["Checks plan and raw evidence"]
    P --> E["Runs one bounded experiment"]
    E --> R["Records result and limitations"]
    R --> D{"Claim gate passed?"}
    D -->|No| P
    D -->|Yes| M["Candidate model card"]
```

| Need | Read | Why |
| --- | --- | --- |
| Recover the original intent | [Original prompt](ORIGINAL_PROMPT.md), [research charter](MISSION.md) | Research temperament, novelty and later corrections. |
| Tell Codex what to do | [Pasteable prompt](PROMPT.md) | Start a fresh session. |
| See the actual state | [Resume brief](RESUME.md) | Evidence, current run and next action. |
| Keep experiments honest | [Working rules](WORK_RULES.md) | Spending gates and evidence standards. |
| Understand the idea | [Plain explanation](../docs/start-here.md) | Reader-friendly goal and diagrams. |
| Check detailed design | [Architecture](../docs/architecture.md), [evaluation](../docs/evaluation.md) | Architecture and claim criteria. |
| Review the experiment | [Forward-pass pilot](../experiments/kaggle/2026-09-27-forward-pilot.md) | Preregistration and eventual verdict. |
| Check first labelled dataset | [CLINC audit](../experiments/data/2026-09-27-clinc-audit.md) | Rights, splits and overlap before model training. |
| Review measured decisions and weaknesses | [CLINC prototype](../experiments/kaggle/2026-09-27-prototype-screen.md), [ContractNLI evidence](../experiments/data/2026-09-27-contractnli-evidence-prototypes.md) | Follow real errors and registered gates. |
| Adversarially review a claimed gain | [Red-team ledger](../docs/red-team.md) | Check shortcuts, missing clauses and deployment gaps. |

When Codex finishes a milestone, update `RESUME.md` and the relevant experiment record with exact run and artifact links, measured outcomes, failures and one concrete next action. Preserve unsuccessful results. Keep this folder concise and link to durable source documents instead of copying them.
