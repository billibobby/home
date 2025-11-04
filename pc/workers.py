"""
Shared Worker Classes for Novita.ai GPU Manager

Provides reusable background worker classes for asynchronous API operations
using Qt's QThreadPool and QRunnable.
"""

import traceback
from PySide6.QtCore import QRunnable, QObject, Signal


class WorkerSignals(QObject):
    """Signals for background worker threads."""
    finished = Signal(object)  # Result data
    error = Signal(tuple)  # (exception_type, exception_value, traceback_str)
    progress = Signal(str)  # Progress message


class APIWorker(QRunnable):
    """Background worker for API operations."""
    
    def __init__(self, fn, *args, **kwargs):
        """
        Initialize worker.
        
        Args:
            fn: Function to execute in background
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
    
    def run(self):
        """Execute the function and emit signals."""
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit((type(e), e, tb_str))

