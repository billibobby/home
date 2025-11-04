# GUI Threading Model and Background Workers

## Overview

The Trading Bot GUI uses a multi-threaded architecture to ensure the UI remains responsive while performing long-running operations. This document explains the threading model and best practices for developers.

## Architecture

### Main Components

1. **Main/GUI Thread**: Runs the Qt event loop and handles all UI updates
2. **Bot Worker Thread**: Executes bot operations (initialization, trading logic)
3. **Thread-Safe Communication**: Uses queues and signals for inter-thread communication

### Design Principles

- **Never block the GUI thread**: All long-running operations run in background threads
- **Thread-safe logging**: Logger is configured once and handlers are thread-safe
- **Signal-based UI updates**: Workers communicate with UI via Qt signals, not direct calls
- **Graceful shutdown**: Stop events allow clean worker termination

## Thread-Safe Logging

### How It Works

The logging system is designed to work safely across multiple threads:

```python
from trading_bot.logger import setup_logger, UILogHandler
import queue

# 1. Set up logger once (called from any thread)
logger = setup_logger('trading_bot', enable_console=False)

# 2. Create a thread-safe queue for GUI logs
log_queue = queue.Queue(maxsize=1000)

# 3. Add UI handler (called from main thread)
ui_handler = UILogHandler(log_queue)
ui_handler.setLevel(logging.INFO)
logger.addHandler(ui_handler)

# 4. Worker threads can now log safely
logger.info("This message goes to the queue")

# 5. GUI polls the queue and updates display
while not log_queue.empty():
    record = log_queue.get_nowait()
    display_in_ui(record.getMessage())
```

### Key Points

- `setup_logger()` is **thread-safe** and can be called from any thread
- `setup_logger()` uses a global cache to prevent duplicate handlers
- File logging is always thread-safe (uses `RotatingFileHandler`)
- Console logging can be disabled with `enable_console=False` for GUI apps
- `UILogHandler` uses `queue.Queue` which is thread-safe
- The GUI must poll the queue from the main thread to update UI

## Background Worker Pattern

### Basic Pattern

```python
import threading
from trading_bot.app import BotApp

class MyGUI:
    def __init__(self):
        self.bot_app = BotApp()
        self.worker_thread = None
    
    def start_bot(self):
        """Start bot in background thread."""
        def worker():
            # This runs in background thread
            self.bot_app.start(blocking=True)
        
        self.worker_thread = threading.Thread(
            target=worker,
            daemon=True  # Dies when main thread exits
        )
        self.worker_thread.start()
    
    def stop_bot(self):
        """Stop bot gracefully."""
        self.bot_app.stop()  # Sets stop event
        
        # Wait for thread to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
```

### Stop Event Pattern

The `BotApp` class uses a threading event for graceful shutdown:

```python
import threading

class BotApp:
    def __init__(self):
        self._stop_event = threading.Event()
    
    def start(self, blocking=True):
        """Run bot until stop() is called."""
        self._stop_event.clear()
        
        # Main bot loop
        while not self._stop_event.is_set():
            # Do work...
            self._stop_event.wait(timeout=1.0)  # Check every second
    
    def stop(self):
        """Signal stop to worker."""
        self._stop_event.set()
```

## GUI Best Practices

### ✓ DO

1. **Initialize bot with console disabled:**
   ```python
   bot_app.initialize(enable_console=False)
   ```

2. **Use daemon threads for background work:**
   ```python
   threading.Thread(target=worker, daemon=True)
   ```

3. **Update UI only from main thread:**
   ```python
   # Use Qt signals
   self.log_signal.emit(message)
   
   # Or use QTimer for polling
   self.timer.timeout.connect(self.process_log_queue)
   ```

4. **Check stop events periodically:**
   ```python
   while not self._stop_event.is_set():
       do_work()
       self._stop_event.wait(timeout=1.0)
   ```

5. **Use timeout when joining threads:**
   ```python
   thread.join(timeout=5.0)
   ```

### ✗ DON'T

1. **Don't update UI from worker threads:**
   ```python
   # BAD - will crash or freeze
   def worker():
       self.label.setText("Working...")  # ✗ Not thread-safe!
   ```

2. **Don't block the GUI thread:**
   ```python
   # BAD - freezes UI
   def on_button_click(self):
       bot_app.start(blocking=True)  # ✗ Blocks GUI!
   ```

3. **Don't call setup_logger() multiple times with same name:**
   ```python
   # BAD - creates duplicate handlers
   for i in range(10):
       setup_logger('trading_bot')  # ✗ Only call once!
   ```

4. **Don't use infinite loops without stop checks:**
   ```python
   # BAD - can't stop gracefully
   def worker():
       while True:  # ✗ No way to exit!
           do_work()
   ```

## Example: Full GUI Integration

Here's a complete example showing proper threading:

```python
import sys
import queue
import threading
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton
from PySide6.QtCore import QTimer, Signal, QObject

from trading_bot.app import BotApp
from trading_bot.logger import UILogHandler


class LogSignals(QObject):
    """Signals for thread-safe log updates."""
    log_message = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create bot and worker
        self.bot_app = BotApp()
        self.worker_thread = None
        
        # Set up log queue and signals
        self.log_queue = queue.Queue(maxsize=1000)
        self.log_signals = LogSignals()
        self.log_signals.log_message.connect(self._append_log)
        
        # Initialize bot (runs in background thread)
        self._init_bot()
        
        # Set up UI
        self._init_ui()
        
        # Start timer for log polling
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self._process_logs)
        self.log_timer.start(100)  # 100ms
    
    def _init_bot(self):
        """Initialize bot in background thread."""
        def init_worker():
            self.bot_app.initialize(enable_console=False)
            
            # Add UI log handler
            logger = self.bot_app.get_logger()
            if logger:
                ui_handler = UILogHandler(self.log_queue)
                ui_handler.setLevel(logging.INFO)
                logger.addHandler(ui_handler)
        
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def _init_ui(self):
        """Set up UI components."""
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        
        # Layout setup...
    
    def _on_start(self):
        """Start bot in background thread."""
        def worker():
            self.bot_app.start(blocking=True)
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def _on_stop(self):
        """Stop bot gracefully."""
        self.bot_app.stop()
    
    def _process_logs(self):
        """Process log queue (runs in main thread)."""
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                message = record.getMessage()
                self.log_signals.log_message.emit(message)
            except queue.Empty:
                break
    
    def _append_log(self, message: str):
        """Append log to display (runs in main thread)."""
        self.log_display.append(message)
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.bot_app:
            self.bot_app.stop()
            self.bot_app.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

## Troubleshooting

### UI Freezes

**Problem**: GUI becomes unresponsive

**Solution**: Move long-running operations to background threads
```python
# Bad
def on_click(self):
    self.bot_app.start(blocking=True)  # Freezes UI!

# Good
def on_click(self):
    thread = threading.Thread(target=lambda: self.bot_app.start(blocking=True))
    thread.start()
```

### Logs Not Appearing

**Problem**: Log messages don't show in GUI

**Solutions**:
1. Ensure `UILogHandler` is added to logger
2. Check that log queue is being polled with QTimer
3. Verify signals are connected: `log_signal.connect(handler)`
4. Check log level: `handler.setLevel(logging.INFO)`

### Can't Stop Bot

**Problem**: Bot doesn't stop when requested

**Solutions**:
1. Ensure worker checks stop event: `if stop_event.is_set(): break`
2. Don't use `while True` without break conditions
3. Use timeouts: `stop_event.wait(timeout=1.0)`
4. Check that `stop()` sets the event: `self._stop_event.set()`

### Duplicate Log Messages

**Problem**: Same message appears multiple times

**Solution**: Don't call `setup_logger()` multiple times
```python
# Bad
for i in range(10):
    logger = setup_logger('trading_bot')  # Creates 10 handlers!

# Good
logger = setup_logger('trading_bot')  # Call once, reuse
```

## References

- [PySide6 Threading Documentation](https://doc.qt.io/qtforpython/overviews/threads-technologies.html)
- [Python Threading Module](https://docs.python.org/3/library/threading.html)
- [Python Queue Module](https://docs.python.org/3/library/queue.html)
- [Python Logging Thread Safety](https://docs.python.org/3/library/logging.html#thread-safety)

