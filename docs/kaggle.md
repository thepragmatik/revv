# Free GPU connection

This project uses Kaggle's notebook service for a small first GPU experiment. GitHub Actions receives the repository secret named `KAGGLE_API_TOKEN` and passes it to the official Kaggle CLI. The workflow checks authentication; it never prints the token or the list of your kernels.

Run the [Kaggle connection check](../.github/workflows/kaggle-connection.yml) in GitHub Actions. A normal push to the research branch that changes this workflow should start the first check. A connector-created commit may not trigger it; GitHub displays **Run workflow** only once the workflow file is on the default branch. A green result means only that the GitHub secret authenticated with the Kaggle API; it does not prove that this account has GPU access, quota, or a measured model result.

The first [Kaggle GPU smoke](../.github/workflows/kaggle-smoke.yml) runs when its workflow file is pushed to the research branch. After the PR is merged, it can also be started manually in GitHub Actions. It submits a **private**, internet-disabled Kaggle script with a five-minute runtime cap, polls for at most twelve minutes, and saves the JSON result as a GitHub Actions artifact. The kernel source is [here](../experiments/kaggle/smoke.py) and the submission script is [here](../analysis/kaggle_smoke.py). The report contains the GPU, execution time, and memory use for a small deterministic computation. This checks availability and result transfer before any model training or dataset download. This number is **not a model latency benchmark**. GPU capacity and quotas can change, so record the machine and date for every result.

The secret belongs in **repository Settings → Secrets and variables → Actions → Repository secrets**. Do not put its value in source files, notebook metadata, issue comments, or Actions logs. If authentication fails, verify that the secret was created in this repository with the exact name `KAGGLE_API_TOKEN` and that the token is an API token from the intended Kaggle account.

See [the roadmap](roadmap.md) for the model experiment stages and [the evaluation contract](evaluation.md) for what counts as a decision model result.
