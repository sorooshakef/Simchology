"""Similarity methods. Use :func:`simchology.list_methods` to see them all."""

from .base import MethodInfo, PairMethod, RepresentationMethod, Runtime, SimilarityMethod
from .registry import get_method_class, list_methods, make_method, register_method, resolve_name

__all__ = [
    "MethodInfo", "PairMethod", "RepresentationMethod", "Runtime", "SimilarityMethod",
    "get_method_class", "list_methods", "make_method", "register_method", "resolve_name",
]
