class PDFTranslatorError(Exception):
    """Base exception for user-facing workflow errors."""


class PDFAnalysisError(PDFTranslatorError):
    """Raised when a PDF cannot be safely analyzed."""


class NoTextLayerError(PDFAnalysisError):
    """Raised when text-only analysis finds no usable PDF text layer."""


class PackageError(PDFTranslatorError):
    """Raised when a translation package cannot be read or matched."""


class ValidationBlockedError(PDFTranslatorError):
    """Raised when output generation is blocked by validation errors."""


class FontNotFoundError(PDFTranslatorError):
    """Raised when no usable Chinese font can be located."""


class InsufficientDiskSpaceError(PDFTranslatorError):
    """Raised when output generation would be unsafe for available disk."""
