# Contributing to Tabero-VTLA

We welcome contributions, improvements, and modifications. Everyone may use Tabero-VTLA under the terms of the [license](LICENSE). Contributors are welcome to submit bug reports, feature requests, and Pull Requests. Because we are a small team with limited review capacity, we cannot guarantee that every PR will be approved, but we will do our best. Detailed guidelines are below.

## Issues and Feature Requests

If you want to discuss topics that are not directly related to bug reports or feature requests, please use GitHub [Discussions](https://github.com/NathanWu7/Tabero-VTLA/discussions). This is suitable for asking how to use specific Tabero-VTLA features or for other general topics.

If you find a bug or another issue, first check whether it has already been reported by using the search bar under GitHub Issues. If the issue has not been reported, please include the following information when opening a GitHub Issue:

- Your operating system type and version, and the Python version you are using
- Code that lets us reproduce the bug, including all dependencies
- The exception traceback
- Any other information that may help us investigate, such as screenshots

To resolve any issue, we must be able to reproduce it. Therefore, if you encounter a problem after modifying Tabero-VTLA, please reproduce the issue without making any additional changes and provide a code snippet that lets us quickly reproduce the problem on the `main` branch.

If you want to submit a feature request, please check whether it already exists and provide the following information:

- The motivation for the feature
- A description of the problem you are trying to solve or your use case
- Enough information for us to understand the nature of the request
- Some information about how you plan to use it, which may help us understand the motivation

We cannot guarantee support for every feature request, but understanding the use cases you care about is very helpful to us.

## Submitting Pull Requests

If you implement support for a new robot or environment, or another new feature, we welcome Pull Requests (PRs) to Tabero-VTLA. We encourage you to first create a [feature request](https://github.com/NathanWu7/Tabero-VTLA/issues) or post on [Discussions](https://github.com/NathanWu7/Tabero-VTLA/discussions) before starting a PR if you want to know whether we are likely to accept it. Because we are a small team with limited maintenance and support capacity, we may not accept all PRs, for example if we think they would make the code harder to maintain or if reviewing the PR is outside our scope. Reaching out early is a good way to understand whether your PR is likely to be merged into Tabero-VTLA. Even if it is not merged, you can of course maintain your own fork and make any changes you want. When creating a PR, we recommend considering the following points for every contribution:

- Make sure your PR has a clear title and description
- Run `pre-commit` (install it first with `pre-commit install`), then run `ruff check .` and `ruff format .`
- Make sure your PR passes all tests
