"""GradEvo: a matched-compute comparison of PPO and NEAT on continuous control.

The package is organized so that the analysis notebook contains no logic: it
imports orchestration from :mod:`gradevo.experiment`, statistics from
:mod:`gradevo.metrics`, and figures from :mod:`gradevo.plots`, then narrates
the results. See ``README.md`` for the research question and hypotheses.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
