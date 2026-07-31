#!/usr/bin/env python3
"""Regression test: update-lockfile checkout must use PAT to push branch."""

import re


def test_update_deps_checkout_pat():
    with open(".github/workflows/_update_dependencies.yml") as f:
        content = f.read()

    # Isolate the update-lockfile job (it precedes create-pr)
    update_job = content.split("  create-pr:")[0]

    match = re.search(
        r"- name: Checkout repo\s+uses: actions/checkout@[^\n]+\s+with:\s+ref: \$\{\{ env\.TARGET_BRANCH \}\}(?:\s+token: \$\{\{ secrets\.PAT \}\})?",
        update_job,
        re.DOTALL,
    )
    assert match is not None, "Could not locate update-lockfile checkout step"

    block = match.group(0)
    assert "token: ${{ secrets.PAT }}" in block, (
        "BUG: update-lockfile checkout must authenticate with PAT to push the bump branch"
    )
