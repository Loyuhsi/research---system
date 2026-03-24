import unittest
import tempfile
from pathlib import Path

from auto_research.runtime import load_config
from conftest import make_temp_repo

class RuntimeTests(unittest.TestCase):
    def test_config_loads_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
            cfg = load_config(repo_root=repo, environ={})
            self.assertEqual(cfg.repo_root, repo)

if __name__ == "__main__":
    unittest.main()
