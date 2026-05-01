"""Tab fragments: Grid, Charts, Schema."""
from __future__ import annotations

from .charts import charts_fragment
from .grid import grid_fragment
from .schema_tab import schema_fragment

__all__ = ["grid_fragment", "charts_fragment", "schema_fragment"]
