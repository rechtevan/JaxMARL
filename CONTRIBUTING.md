# Contributing to JaxMARL

Please help build JaxMARL into the best possible tool for the MARL community.

## Contributing code

We actively welcome your contributions!

- If adding an environment or algorithm, check with us that it is the right fit for the
  repo.
- Fork the repo and create your branch from main.
- Add tests, or show proof that the environment/algorithm works. The exact requirements
  are listed below.
- Add a README explaining your environment/algorithm.

**Environment Requirements**

- Unit tests (in `pytest` format) demonstrating correctness. If applicable, show
  correspondence to existing implementations. If transitions match, write a unit test to
  demonstrate this
  ([example](https://github.com/FLAIROx/JaxMARL/blob/be9fe46e52a736f8dd766acf98b4e0803f199dd2/tests/mpe/test_mpe.py)).
- Training results for IPPO and MAPPO over 20 seeds, with configuration files saved to
  `baselines`.

**Algorithm Requirements**

- Performance results on at least 3 environments (e.g. SMAX, MABrax & Overcooked) with
  at least 20 seeds per result.
- If applicable, compare performance results to existing implementations to demonstrate
  correctness.

## Bug reports

We use Github's issues to track bugs, just open a new issue! Great Bug Reports tend to
have:

- A quick summary and/or background
- Steps to reproduce (Be specific and give example code if you can)
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried
  that didn't work)

## License

**JaxMARL is licensed under the Apache License 2.0.**

By contributing to this fork (rechtevan/JaxMARL), you agree that:

- All contributions will be licensed under the Apache License 2.0
- Code contributed to this fork is **Copyright Booz Allen Hamilton**
- All code contributions (source code, tests, documentation) must be compatible with
  Apache 2.0
- Contributions can be freely used, modified, and distributed under Apache 2.0 terms

**Note**: Contributions to this fork acknowledge Booz Allen Hamilton copyright while
maintaining Apache 2.0 license terms.

### Dependency Licensing Requirements

When adding new dependencies:

- **Runtime dependencies** must use Apache 2.0, MIT, BSD, or other permissive licenses
- **Development dependencies** (testing, linting, etc.) can use any OSI-approved license
- **Avoid GPL/LGPL** licenses as they may conflict with Apache 2.0 distribution
- Document any new dependencies in `pyproject.toml` with appropriate license classifiers

### Acceptable Licenses for Dependencies

✅ **Recommended (most permissive):**

- Apache License 2.0
- MIT License
- BSD License (2-clause, 3-clause)
- ISC License

✅ **Acceptable for dev dependencies only:**

- Any OSI-approved license for tools like pytest, ruff, mypy, etc.

❌ **Not acceptable for runtime dependencies:**

- GPL, LGPL, AGPL (copyleft licenses)
- Proprietary or custom restrictive licenses

If you're unsure about a dependency's license compatibility, ask before adding it!

### Copyright Headers and Attribution

**POLICY: Industry Standard Approach (Option A)**

This repository follows industry standard practices for multi-company Apache 2.0
projects:

✅ **NOTICE file** - Lists major contributors (FLAIR, Booz Allen Hamilton) ✅ **Git
commits** - Provide detailed attribution (use @bah.com email) ✅ **LICENSE file** -
Apache 2.0 covers all code ❌ **NO copyright headers** - Not required, matches industry
standard

This is the same approach used by Kubernetes, TensorFlow, Linux, Apache Spark, and
hundreds of other multi-company projects.

#### Default Approach: NO HEADERS NEEDED

When contributing to this fork:

**The general rule: Do NOT add copyright headers**

- The LICENSE + NOTICE files provide legal coverage
- Git commit history provides attribution
- Use your @bah.com email in git commits
- This maintains consistency with industry standard practice
- Copyright headers are optional, not required

**For most contributions:**

- Modify existing files → No header needed
- Create new test files → No header needed
- Create new utility files → No header needed
- Git history tracks your contribution

**To see who contributed to a file:**

```bash
git log --format="%an <%ae>" <filename> | sort | uniq -c
```

#### Exception: When Headers ARE Required

Only add copyright headers if:

1. Your organization requires it for legal/compliance
2. Creating substantial new components (new algorithms, environments)
3. File already has copyright headers (preserve existing, then append yours)

**For substantial NEW files** (if headers required):

```python
# Copyright 2025 Booz Allen Hamilton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

**For files WITH existing headers** (preserve and append):

```python
# Copyright 2023 Original Author
# Copyright 2025 Booz Allen Hamilton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

**Remember:** When in doubt, omit headers. The LICENSE + NOTICE files and git history
provide full legal coverage and attribution.

### Optional: DCO Sign-Off (Best Practice)

Many major open source projects use **Developer Certificate of Origin (DCO)** sign-offs
to certify contribution rights.

**To enable DCO sign-off on your commits:**

```bash
# Configure git with your Booz Allen Hamilton email
git config user.name "Your Name"
git config user.email "your.name@bah.com"

# Commit with sign-off
git commit -s -m "Add coverage tracking workflow"
```

This adds:

```text
Signed-off-by: Your Name <your.name@bah.com>
```

**What DCO certifies:**

- You created the contribution OR have rights to submit it
- You're submitting under the project's license (Apache 2.0)
- You understand the public nature of the contribution

**DCO is optional but recommended** - used by Linux, Kubernetes, Docker, and many
others.

## Roadmap

Some improvements we would like to see implemented:

- [x] improved RNN implementations. In the current implementation, the hidden size is
  dependent on "NUM_STEPS", it should be made independent.
- [ ] S5 RNN architecture.
