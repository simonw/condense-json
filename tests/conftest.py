import os

from hypothesis import settings

# HYPOTHESIS_PROFILE=thorough runs the property tests with 20x the
# examples - worth doing before a release, too slow for every run.
settings.register_profile("thorough", max_examples=2000)
settings.register_profile("default", max_examples=100)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
