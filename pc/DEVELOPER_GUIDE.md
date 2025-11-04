# Developer Guide - Novita.ai GPU Manager

## Quick Reference for Phase 3+ Development

---

## 🏗️ Architecture Overview

### Component Hierarchy

```
main.py (Entry Point)
    ↓
    ├── ConfigManager (config_manager.py)
    │   └── JSON file: ~/.novita_gpu_manager/config.json
    │
    ├── NovitaAPIClient (novita_api.py)
    │   └── Session with Bearer token authentication
    │
    └── MainWindow (main_window.py)
        ├── SettingsDialog (settings_dialog.py)
        │   └── Uses ConfigManager
        │
        ├── Instance Table (QTableWidget)
        ├── Activity Logs (QTextEdit)
        ├── Control Buttons
        └── Menu/Toolbar/StatusBar
```

---

## 📂 File Responsibilities

| File | Responsibility | Key Classes/Functions |
|------|----------------|----------------------|
| `main.py` | Application bootstrap | `main()` |
| `config_manager.py` | Settings persistence | `ConfigManager` |
| `settings_dialog.py` | Settings UI | `SettingsDialog` |
| `main_window.py` | Main UI | `MainWindow` |
| `novita_api.py` | API communication | `NovitaAPIClient` |

---

## 🔌 Key APIs and Interfaces

### ConfigManager API

```python
config = ConfigManager()

# Read
value = config.get('api_key', default='')
all_config = config.get_all()

# Write
config.set('api_key', 'new_key')
config.update({'key1': 'value1', 'key2': 'value2'})
config.save()  # Persist to disk

# Auto-loads on init, auto-creates directory
```

**Config Keys:**
- `api_key` (str)
- `default_gpu_product_id` (str)
- `default_cluster_id` (str)
- `docker_image_url` (str)
- `window_geometry` (dict: x, y, width, height)
- `window_state` (bytes/None)

### NovitaAPIClient API

```python
client = NovitaAPIClient(api_key)

# Existing methods (Phase 1+2)
user_info = client.get_user_info()
products = client.list_gpu_products()
clusters = client.list_clusters()

# TODO Phase 3: Add these methods
# instances = client.list_instances()
# instance = client.create_instance(...)
# client.start_instance(instance_id)
# client.stop_instance(instance_id)
# client.delete_instance(instance_id)
# status = client.get_instance_status(instance_id)
```

**Exception Handling:**
```python
from novita_api import NovitaAPIError, NovitaAuthenticationError

try:
    result = client.some_api_call()
except NovitaAuthenticationError:
    # Invalid API key
    pass
except NovitaAPIError as e:
    # Other API errors
    pass
```

### MainWindow API

```python
# Logging
self.log_message("Operation completed", "SUCCESS")
self.log_message("Warning message", "WARNING")
self.log_message("Error occurred", "ERROR")
self.log_message("Info message", "INFO")

# Instance management (Phase 3)
self.instances = []  # List of instance dicts
self.update_instance_count()

# Status bar updates
self.status_bar.showMessage("Operation complete", 3000)  # 3s timeout
self.api_status_label.setText("API: Connected")
self.credit_label.setText(f"Credits: ${credits:.2f}")

# Table operations
self.instance_table.setRowCount(len(instances))
# ... populate rows ...
```

---

## 🎯 Phase 3 Implementation Guide

### Adding Instance Creation

**1. Update `novita_api.py`:**

```python
def create_instance(self, 
                    gpu_product_id: str,
                    cluster_id: str = None,
                    docker_image: str = None,
                    name: str = None) -> Dict[str, Any]:
    """Create new GPU instance."""
    payload = {
        "gpu_product_id": gpu_product_id,
        # ... other params
    }
    
    response = self.session.post(
        f"{self.BASE_URL}/instances",
        json=payload
    )
    response.raise_for_status()
    return response.json()
```

**2. Create `create_instance_dialog.py`:**

```python
from PyQt5.QtWidgets import QDialog, ...
from config_manager import ConfigManager

class CreateInstanceDialog(QDialog):
    def __init__(self, config_manager, api_client, parent=None):
        super().__init__(parent)
        # Load GPU products from API
        # Populate combo boxes
        # Get defaults from config
        # ...
```

**3. Update `main_window.py`:**

Replace placeholder method:

```python
def _create_instance(self):
    """Open create instance dialog and create instance."""
    try:
        dialog = CreateInstanceDialog(
            self.config_manager,
            self.api_client,
            self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            # Get instance params from dialog
            params = dialog.get_instance_params()
            
            # Create instance via API
            self.log_message("Creating instance...", "INFO")
            instance = self.api_client.create_instance(**params)
            
            # Add to local list
            self.instances.append(instance)
            
            # Refresh table
            self._refresh_instances()
            
            self.log_message(
                f"Instance created: {instance['name']}",
                "SUCCESS"
            )
            
    except Exception as e:
        self.log_message(f"Failed to create instance: {e}", "ERROR")
        QMessageBox.critical(self, "Error", str(e))
```

**4. Update button connection:**

In `_create_instance_area()`, change:
```python
self.create_btn.clicked.connect(self._create_instance)
```

### Adding Instance Refresh

**1. Update `novita_api.py`:**

```python
def list_instances(self) -> List[Dict[str, Any]]:
    """List all GPU instances."""
    response = self.session.get(f"{self.BASE_URL}/instances")
    response.raise_for_status()
    data = response.json()
    return data.get("instances", [])
```

**2. Update `main_window.py`:**

```python
def _refresh_instances(self):
    """Fetch and display instances from API."""
    try:
        self.log_message("Refreshing instances...", "INFO")
        
        # Fetch from API
        instances = self.api_client.list_instances()
        self.instances = instances
        
        # Update table
        self._populate_instance_table()
        
        # Update count
        self.update_instance_count()
        
        self.status_bar.showMessage(
            f"Refreshed: {len(instances)} instances",
            3000
        )
        
    except Exception as e:
        self.log_message(f"Refresh failed: {e}", "ERROR")

def _populate_instance_table(self):
    """Populate table with self.instances data."""
    table = self.instance_table
    
    if not self.instances:
        # Show placeholder
        table.setRowCount(1)
        item = QTableWidgetItem("No instances...")
        item.setFlags(Qt.ItemIsEnabled)
        table.setItem(0, 0, item)
        table.setSpan(0, 0, 1, 6)
        return
    
    # Clear placeholder
    table.setRowCount(0)
    table.clearSpans()
    
    # Populate rows
    for i, instance in enumerate(self.instances):
        table.insertRow(i)
        table.setItem(i, 0, QTableWidgetItem(instance.get('name', '')))
        table.setItem(i, 1, QTableWidgetItem(instance.get('id', '')))
        table.setItem(i, 2, QTableWidgetItem(instance.get('status', '')))
        # ... other columns
```

---

## 🎨 UI Conventions

### Color Coding

**Log Messages:**
- INFO: black
- WARNING: orange  
- ERROR: red
- SUCCESS: green

**Status Indicators:**
- API Connected: green
- API Disconnected: red

### Message Timing

```python
# Permanent message
self.status_bar.showMessage("Ready")

# Temporary message (3 seconds)
self.status_bar.showMessage("Saved", 3000)
```

### Confirmation Dialogs

```python
reply = QMessageBox.question(
    self,
    "Confirm Delete",
    f"Delete instance '{name}'?",
    QMessageBox.Yes | QMessageBox.No,
    QMessageBox.No  # Default
)

if reply == QMessageBox.Yes:
    # Proceed with deletion
```

---

## 🔍 Debugging Tips

### Enable Debug Logging

In `main.py`:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Inspect Config File

```python
# Config location
import pathlib
config_path = pathlib.Path.home() / ".novita_gpu_manager" / "config.json"
print(config_path)
```

### Test API Calls

```python
# In Python shell
from novita_api import NovitaAPIClient
client = NovitaAPIClient("your_key")
print(client.get_user_info())
```

### UI Testing Without API

```python
# In main.py, create mock client
class MockAPIClient:
    def get_user_info(self):
        return {"credits": 100.0, "username": "test"}
    
    def list_instances(self):
        return [
            {"id": "1", "name": "Test Instance", "status": "running"},
        ]

# Use mock in development
api_client = MockAPIClient()
```

---

## 📦 Adding New Dependencies

1. Install package:
   ```bash
   pip install package-name
   ```

2. Update `requirements.txt`:
   ```bash
   pip freeze | grep package-name >> requirements.txt
   ```

3. Document in README

---

## 🧪 Testing Guidelines

### Unit Tests (Future)

Create `tests/` directory:
```
tests/
├── test_config_manager.py
├── test_novita_api.py
└── test_main_window.py
```

### Manual Testing Checklist

Before committing new features:
- [ ] Feature works with valid inputs
- [ ] Feature handles invalid inputs gracefully
- [ ] Error messages are user-friendly
- [ ] Logs show appropriate messages
- [ ] UI remains responsive during operations
- [ ] Config persists correctly
- [ ] No crashes or exceptions

---

## 🔐 Security Considerations

### API Key Handling

- ✅ Stored in user-only readable file (0o600)
- ✅ Never logged in plain text
- ✅ Masked in UI by default
- ⚠️ NOT encrypted (acceptable for local app)

### Future Improvements

- Keyring integration for secure storage
- API key expiration checking
- Rate limit handling

---

## 📚 Resources

### PyQt5 Documentation
- [PyQt5 Reference](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [Qt Widgets](https://doc.qt.io/qt-5/qtwidgets-module.html)

### Python Best Practices
- [PEP 8 Style Guide](https://pep8.org/)
- [Type Hints](https://docs.python.org/3/library/typing.html)

### Novita.ai
- [API Documentation](https://novita.ai/docs)
- [Dashboard](https://novita.ai/dashboard)

---

## 🤝 Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Log important operations
- Handle exceptions gracefully

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/instance-management

# Make changes
git add .
git commit -m "Add instance creation dialog"

# Push and create PR
git push origin feature/instance-management
```

### Commit Message Format

```
Add instance creation functionality

- Implement CreateInstanceDialog
- Add create_instance API method
- Wire up UI buttons
- Add error handling
```

---

## 🐛 Common Issues

### "QWidget: Must construct a QApplication before a QWidget"

Create QApplication before any widgets:
```python
app = QApplication(sys.argv)
window = MainWindow(...)  # Now OK
```

### "Config file permission denied"

Check file permissions:
```bash
ls -la ~/.novita_gpu_manager/config.json
chmod 600 ~/.novita_gpu_manager/config.json
```

### "API 401 Unauthorized"

- Check API key is correct
- Verify key has not expired
- Test key on Novita.ai dashboard

---

**Last Updated**: Phase 2 Completion  
**Maintainer**: Development Team  
**Status**: Living Document

