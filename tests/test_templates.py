import unittest

from omniloader import OmniConfig, config_template_path
from omniloader.config import COLLATES, STRATEGIES
from omniloader.data.factory import ADAPTERS
from omniloader.transforms import TRANSFORMS


class TestConfigTemplate(unittest.TestCase):
    def test_path_points_to_existing_file(self):
        path = config_template_path()
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix, ".yaml")

    def test_template_loads_as_config(self):
        # The shipped template must be a valid, loadable OmniConfig.
        config = OmniConfig.from_file(config_template_path())
        self.assertIsInstance(config, OmniConfig)
        self.assertEqual(config.strategy, "temperature")
        self.assertTrue(config.pad_features)
        self.assertEqual(len(config.datasets), 2)

    def test_referenced_names_are_registered(self):
        # Every name the template names must exist in its registry, so the
        # template can never drift into referencing a removed option.
        config = OmniConfig.from_file(config_template_path())
        self.assertIn(config.strategy, STRATEGIES)
        self.assertIn(config.collate, COLLATES)
        for entry in config.datasets:
            self.assertIn(entry["adapter"], ADAPTERS)
        for transform in config.transforms:
            self.assertIn(transform["name"], TRANSFORMS)


if __name__ == "__main__":
    unittest.main()
