# Contributing To Export-Deploy

Thanks for your interest in contributing to Export-Deploy!

## Setting Up

### Development Container

1. **Build and run the Docker container**:

```bash
docker build -f docker/Dockerfile.ci -t Export-Deploy .
```

```bash
docker run --rm -it --entrypoint bash --runtime nvidia --gpus all Export-Deploy
```

### Development Dependencies

We use [uv](https://docs.astral.sh/uv/) for managing dependencies.

New required dependencies can be added by `uv add $DEPENDENCY`.

New optional dependencies can be added by `uv add --optional --extra $EXTRA $DEPENDENCY`.

`EXTRA` refers to the subgroup of extra-dependencies to which you're adding the new dependency.
Example: For adding a TRT-LLM specific dependency, run `uv add --optional --extra trtllm $DEPENDENCY`.

Adding a new dependency will update UV's lock-file. Please check this into your branch:

```bash
git add uv.lock pyproject.toml
git commit -m "build: Adding dependencies"
git push
```

### Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

Installation:

```bash
pip install ruff
```

Format:

```bash
ruff format .
```

## Making Changes

### Workflow: Clone and Branch (No Fork Required)

Pre-commit checks will help ensure your code follows our formatting and style guidelines.

We follow a direct clone and branch workflow for now:

1. Clone the repository directly:

   ```bash
   git clone https://github.com/NVIDIA-NeMo/Export-Deploy
   cd Export-Deploy
   ```

2. Create a new branch for your changes:

   ```bash
   git checkout -b your-feature-name
   ```

3. Make your changes and commit them:

   ```bash
   git add .
   git commit --signoff -m "Your descriptive commit message"
   ```

We require signing commits with `--signoff` (or `-s` for short). See [Signing Your Work](#signing-your-work) for details.

4. Push your branch to the repository:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a pull request from your branch to the `main` branch.

### Design Documentation Requirement

**Important**: All new key features (ex: enabling a new inference optimized library, enabling a new deployment option) must include documentation update (either a new doc or updating an existing one). This document update should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

This ensures that all significant changes are well-thought-out and properly documented for future reference. Comprehensive documentation serves two critical purposes:

1. **User Adoption**: Helps users understand how to effectively use the library's features in their projects
2. **Developer Extensibility**: Enables developers to understand the internal architecture and implementation details, making it easier to modify, extend, or adapt the code for their specific use cases

Quality documentation is essential for both the usability of Export-Deploy and its ability to be customized by the community.

## Code Quality

- Follow the existing code style and conventions
- Write tests for new features
- Update documentation to reflect your changes
- Ensure all tests pass before submitting a PR
- Do not add arbitrary defaults for configs, be as explicit as possible.

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
