#!/usr/bin/env python3
import sys
from unittest.mock import MagicMock

class MockFinder:
    @classmethod
    def find_spec(cls, name, path, target=None):
        if name.startswith("pygments"):
            from importlib.machinery import ModuleSpec
            spec = ModuleSpec(name, loader=MockLoader)
            spec.submodule_search_locations = []
            return spec
        return None

class MockLoader:
    @classmethod
    def create_module(cls, spec):
        m = MagicMock()
        m.__spec__ = spec
        m.__path__ = []
        return m

    @classmethod
    def exec_module(cls, module):
        pass

sys.meta_path.insert(0, MockFinder)

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
