"""OmniKV error hierarchy.

Design rule (research.md §7): unsupported or unverified combinations fail
fast with actionable messages; there is no silent fallback.
"""


class OmniKVError(Exception):
    """Base class for all OmniKV errors."""


class LayoutError(OmniKVError):
    """A cache layout could not be described, certified, or trusted."""


class CompatibilityError(OmniKVError):
    """Producer and consumer formats are incompatible and no codec exists."""


class RegistrationError(OmniKVError):
    """The engine's caches could not be registered (uncertifiable group)."""


class OwnershipError(OmniKVError):
    """A block ownership/epoch invariant was violated."""


class StoreError(OmniKVError):
    """A storage backend operation failed."""
