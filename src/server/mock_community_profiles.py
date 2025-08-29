"""
Mock Community Profiles for SIRAJ v6.1
Provides a set of mock community profiles for simulating user-driven validation.
"""

from typing import Dict, List, Any

class MockCommunityProfiles:
    """Manages mock community profiles."""

    def __init__(self):
        self.profiles = {
            "linguistic_scholar": {
                "name": "Linguistic Scholar",
                "description": "Focuses on linguistic accuracy and etymological rigor.",
                "preferences": {
                    "linguistic_accuracy": 1.5,
                    "cultural_appropriateness": 0.8,
                    "computational_validity": 1.0
                }
            },
            "cultural_keeper": {
                "name": "Cultural Keeper",
                "description": "Prioritizes cultural appropriateness and traditional knowledge.",
                "preferences": {
                    "linguistic_accuracy": 0.8,
                    "cultural_appropriateness": 2.0,
                    "computational_validity": 0.5
                }
            },
            "computational_expert": {
                "name": "Computational Expert",
                "description": "Emphasizes computational validity and statistical significance.",
                "preferences": {
                    "linguistic_accuracy": 1.0,
                    "cultural_appropriateness": 0.7,
                    "computational_validity": 1.8
                }
            },
            "community_representative": {
                "name": "Community Representative",
                "description": "Balances all aspects of validation to represent the community's interests.",
                "preferences": {
                    "linguistic_accuracy": 1.2,
                    "cultural_appropriateness": 1.5,
                    "computational_validity": 1.0
                }
            }
        }

    def get_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all mock community profiles."""
        return self.profiles

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """Get a specific mock community profile."""
        return self.profiles.get(profile_id)
