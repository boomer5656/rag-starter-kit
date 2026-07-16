"""Source connectors — each yields SourceDocs for the pipeline to process.

Every connector implements the `Connector` protocol (base.py): a
`source_type` tag, an `index_fields` map for Qdrant payload indexes, and
`iter_docs()`. Add a new source by writing one of these, not by touching
the pipeline.
"""
from .base import Connector
from .files import FilesConnector
from .literature import LiteratureConnector
from .sigma import SigmaConnector

__all__ = ["Connector", "FilesConnector", "LiteratureConnector", "SigmaConnector"]
