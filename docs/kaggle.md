# Free GPU connection

This project uses Kaggle's notebook service for a small first GPU experiment. GitHub Actions receives the repository secret named `KAGGLE_API_TOKEN` and passes it to the official Kaggle CLI. The workflow checks authentication; it never prints the token or the list of your kernels.

Run the [Kaggle connection check](../.github/workflows/kaggle-connection.yml) in GitHub Actions. A push to the research branch that changes this workflow starts the first check. After the workflow reaches the default branch, you can start it using **Run workflow**. A green result means only that the GitHub secret authenticated with the Kaggle API; it does not prove that this account has GPU access, quota, or a measured model result.

Next, run a bounded synthetic GPU smoke test with a private Kaggle script. It will report device, execution time, maximum GPU memory, and a small deterministic computation. This will verify availability and the result transfer before any model training or dataset download. GPU capacity and quotas can change, so record the machine and date for every result.

The secret belongs in **repository Settings → Secrets and variables → Actions → Repository secrets**. Do not put its value in source files, notebook metadata, issue comments, or Actions logs. If authentication fails, verify that the secret was created in this repository with the exact name `KAGGLE_API_TOKEN` and that the token is an API token from the intended Kaggle account.

See [the roadmap](roadmap.md) for the model experiment stages and [the evaluation contract](evaluation.md) for what counts as a decision model result.
