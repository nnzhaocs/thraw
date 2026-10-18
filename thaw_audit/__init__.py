"""The ledger that decides what has already been settled.

Every number in the paper traces to a field of a results file. `semview.runlog`
already records how a file was produced -- a run directory, a command, an
environment, a manifest, a line in `RUNS.md`. What was missing is the other
half: a statement of which numbers are SETTLED, so that checking them is a
comparison rather than an experiment.

Without that half, an auditor -- human or otherwise -- has no stop condition.
It reads that a figure has no run directory, re-measures it, gets a third value,
and now has more to reconcile than it started with. The loop is not a mistake in
the auditing; it is the absence of a thing to audit against.

So this package adds exactly one idea: a claim can be **frozen**. A frozen
claim's source file hash is recorded, and verifying it means checking that hash.
If it matches, the claim is settled and nothing else happens -- no field is
re-read, no command is run, no number is recomputed. If it does not match, that
is an error loud enough to need a person, because a file changed underneath a
number the paper already states.

The three modes exist for the same reason. `AUDIT` cannot execute anything, so
an audit can never turn into a measurement halfway through; `MEASURE` can, but
has to register the run first; `IMPLEMENT` touches code and is forbidden from
writing to the ledger at all.
"""

from .claims import Claim, Registry, STATUSES          # noqa: F401
from .hashes import sha256_cached                      # noqa: F401
from .ledger import Ledger                             # noqa: F401
from .modes import Mode, ModeError, require            # noqa: F401
from .paths import resolve                             # noqa: F401
from .verify import verify_claim, verify_all           # noqa: F401

__all__ = ["Claim", "Registry", "STATUSES", "Ledger", "Mode", "ModeError",
           "require", "resolve", "sha256_cached", "verify_claim", "verify_all"]
