#!/usr/bin/env python3
"""Regression test: is_optional description matches behavior."""

import sys


def main():
    with open(".github/actions/test-template/action.yml") as f:
        content = f.read()

    if "Failure will cancel all other tests if set to true" in content:
        print("BUG: is_optional description inverts the actual behavior")
        sys.exit(1)

    if "is_optional:" not in content or "description:" not in content:
        print("Could not find is_optional input description")
        sys.exit(1)

    print("OK")


