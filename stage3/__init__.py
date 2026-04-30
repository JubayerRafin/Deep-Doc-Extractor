"""
Stage 3 — Dataset Augmentation & Final Export
Exports the main classes and orchestrator function for use by pipeline.py.

Team DocuForge | Woosong University Capstone 2026
Mentor: Mintae Kim (HP Korea)
"""

from .llm_rephraser import LLMRephraser, rephrase_question
from .augment_filter import AugmentFilter, filter_variations
from .final_exporter import FinalExporter, export_augmented_dataset
from .stage3_pipeline import run_pipeline

__all__ = [
    "LLMRephraser",
    "AugmentFilter",
    "FinalExporter",
    "rephrase_question",
    "filter_variations",
    "export_augmented_dataset",
    "run_pipeline",
]
