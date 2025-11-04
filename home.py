"""
Home Launcher - Unified Application Manager

A GUI launcher for managing and launching AI Trading Bot and GPU Manager applications.
Provides a single entry point for multiple desktop applications.
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QWidget,
    QMessageBox,
)
from PySide6.QtCore import Qt, QProcess, QTimer, QProcessEnvironment
from PySide6.QtGui import QFont, QIcon, QGuiApplication


class HomeWindow(QMainWindow):
    """Main window for the Home Launcher application."""
    
    def __init__(self):
        super().__init__()
        self.bot_process = None
        self.gpu_process = None
        self.init_ui()
        self.apply_styling()
    
    def init_ui(self):
        """Initialize the user interface components."""
        # Set window properties
        self.setWindowTitle("Home Launcher")
        self.setFixedSize(500, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title label
        title_label = QLabel("Application Launcher")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle label
        subtitle_label = QLabel("Select an application to launch")
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #666666;")
        layout.addWidget(subtitle_label)
        
        # Add spacing
        layout.addStretch()
        
        # Bot launch button
        self.bot_button = QPushButton("🤖 Launch AI Trading Bot")
        self.bot_button.setMinimumHeight(80)
        self.bot_button.setFont(QFont("Arial", 14))
        self.bot_button.setToolTip("Launch the AI Trading Bot with XGBoost ML trading system")
        self.bot_button.clicked.connect(self.launch_bot)
        layout.addWidget(self.bot_button)
        
        # GPU Manager launch button
        self.gpu_button = QPushButton("💻 Launch GPU Manager")
        self.gpu_button.setMinimumHeight(80)
        self.gpu_button.setFont(QFont("Arial", 14))
        self.gpu_button.setToolTip("Launch the Novita.ai GPU Instance Manager")
        self.gpu_button.clicked.connect(self.launch_gpu_manager)
        layout.addWidget(self.gpu_button)
        
        # Add spacing
        layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Ready to launch")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        layout.addWidget(self.status_label)
    
    def apply_styling(self):
        """Apply modern styling to the window and buttons."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#gpu_button {
                background-color: #2196F3;
            }
            QPushButton#gpu_button:hover {
                background-color: #0b7dda;
            }
            QPushButton#gpu_button:pressed {
                background-color: #0a6bc2;
            }
        """)
        # Set object name for GPU button for specific styling
        self.gpu_button.setObjectName("gpu_button")
    
    def center_window(self):
        """Center the window on the screen."""
        screen = QGuiApplication.primaryScreen().availableGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)
    
    def showEvent(self, event):
        """Handle window show event - center window after it's shown."""
        super().showEvent(event)
        self.center_window()
    
    def get_bot_executable_path(self):
        """Get the path to the bot executable if it exists."""
        repo_root = Path(__file__).parent
        exe_path = repo_root / "bot" / "dist" / "TradingBotGUI.exe"
        if exe_path.exists() and exe_path.is_file():
            return exe_path
        return None
    
    def get_gpu_manager_executable_path(self):
        """Get the path to the GPU Manager executable if it exists."""
        repo_root = Path(__file__).parent
        exe_path = repo_root / "pc" / "dist" / "GPUManager.exe"
        if exe_path.exists() and exe_path.is_file():
            return exe_path
        return None
    
    def launch_bot(self):
        """Launch the AI Trading Bot application."""
        try:
            self.status_label.setText("Launching AI Trading Bot...")
            self.status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.update()
            
            repo_root = Path(__file__).parent
            bot_dir = repo_root / "bot"
            
            # Create QProcess instance
            if self.bot_process:
                self.bot_process.kill()
                self.bot_process.waitForFinished(1000)
            self.bot_process = QProcess(self)
            
            # Connect signals for better error handling
            self.bot_process.errorOccurred.connect(
                lambda error: self._handle_process_error(self.bot_process, "AI Trading Bot", error)
            )
            self.bot_process.finished.connect(
                lambda exit_code, exit_status: self._handle_process_finished(
                    self.bot_process, "AI Trading Bot", exit_code, exit_status
                )
            )
            
            # Priority 1: Check for executable
            exe_path = self.get_bot_executable_path()
            if exe_path:
                # Launch executable
                self.bot_process.setWorkingDirectory(str(bot_dir))
                self.bot_process.start(str(exe_path))
            else:
                # Priority 2: Try direct script execution (primary dev-mode path)
                script_path = bot_dir / "src" / "trading_bot" / "gui_main.py"
                if script_path.exists():
                    # Set environment with PYTHONPATH
                    env = QProcessEnvironment.systemEnvironment()
                    bot_src_path = str(bot_dir / "src")
                    pythonpath = env.value("PYTHONPATH", "")
                    if pythonpath:
                        env.insert("PYTHONPATH", bot_src_path + os.pathsep + pythonpath)
                    else:
                        env.insert("PYTHONPATH", bot_src_path)
                    self.bot_process.setProcessEnvironment(env)
                    self.bot_process.setWorkingDirectory(str(bot_dir))
                    self.bot_process.start(sys.executable, [str(script_path)])
                else:
                    # Priority 3: Try module execution with PYTHONPATH
                    # Set environment with PYTHONPATH
                    env = QProcessEnvironment.systemEnvironment()
                    bot_src_path = str(bot_dir / "src")
                    pythonpath = env.value("PYTHONPATH", "")
                    if pythonpath:
                        env.insert("PYTHONPATH", bot_src_path + os.pathsep + pythonpath)
                    else:
                        env.insert("PYTHONPATH", bot_src_path)
                    self.bot_process.setProcessEnvironment(env)
                    self.bot_process.setWorkingDirectory(str(bot_dir))
                    self.bot_process.start(sys.executable, ["-m", "trading_bot.gui_main"])
            
            # Schedule error check after short delay
            QTimer.singleShot(1000, lambda proc=self.bot_process: self._check_process_status(proc, "AI Trading Bot"))
            
            self.status_label.setText("AI Trading Bot started")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        except OSError as e:
            error_msg = f"Launch Error: {str(e)}"
            QMessageBox.critical(self, "Launch Error", error_msg)
            self.status_label.setText(f"Error: Launch failed")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            self.bot_process = None
        
        except FileNotFoundError as e:
            error_msg = f"Executable or script not found.\n\n{str(e)}\n\nPlease build the application first or check installation."
            QMessageBox.critical(self, "Launch Error", error_msg)
            self.status_label.setText(f"Error: File not found")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        except Exception as e:
            error_msg = f"Unexpected error occurred.\n\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
    
    def launch_gpu_manager(self):
        """Launch the GPU Manager application."""
        try:
            self.status_label.setText("Launching GPU Manager...")
            self.status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.update()
            
            repo_root = Path(__file__).parent
            pc_dir = repo_root / "pc"
            
            # Create QProcess instance
            if self.gpu_process:
                self.gpu_process.kill()
                self.gpu_process.waitForFinished(1000)
            self.gpu_process = QProcess(self)
            
            # Connect signals for better error handling
            self.gpu_process.errorOccurred.connect(
                lambda error: self._handle_process_error(self.gpu_process, "GPU Manager", error)
            )
            self.gpu_process.finished.connect(
                lambda exit_code, exit_status: self._handle_process_finished(
                    self.gpu_process, "GPU Manager", exit_code, exit_status
                )
            )
            
            # Priority 1: Check for executable
            exe_path = self.get_gpu_manager_executable_path()
            if exe_path:
                # Launch executable
                self.gpu_process.setWorkingDirectory(str(pc_dir))
                self.gpu_process.start(str(exe_path))
            else:
                # Priority 2: Try Python script
                script_path = pc_dir / "main.py"
                if script_path.exists():
                    self.gpu_process.setWorkingDirectory(str(pc_dir))
                    self.gpu_process.start(sys.executable, [str(script_path)])
                else:
                    raise FileNotFoundError(
                        f"GPU Manager executable not found: {exe_path}\n"
                        f"GPU Manager script not found: {script_path}\n"
                        "Please build the application or ensure Python scripts are available."
                    )
            
            # Schedule error check after short delay
            QTimer.singleShot(1000, lambda proc=self.gpu_process: self._check_process_status(proc, "GPU Manager"))
            
            self.status_label.setText("GPU Manager started")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        except OSError as e:
            error_msg = f"Launch Error: {str(e)}"
            QMessageBox.critical(self, "Launch Error", error_msg)
            self.status_label.setText(f"Error: Launch failed")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            self.gpu_process = None
        
        except FileNotFoundError as e:
            error_msg = f"Executable or script not found.\n\n{str(e)}\n\nPlease build the application first or check installation."
            QMessageBox.critical(self, "Launch Error", error_msg)
            self.status_label.setText(f"Error: File not found")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        except Exception as e:
            error_msg = f"Unexpected error occurred.\n\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
    
    def _check_process_status(self, process, app_name):
        """Check if a process has crashed immediately after launch."""
        if process is None:
            return
        
        if process.state() == QProcess.ProcessState.NotRunning:
            exit_code = process.exitCode()
            if exit_code != 0:
                error_output = process.readAllStandardError().data().decode('utf-8', errors='ignore')
                error_msg = f"{app_name} failed to start.\n\n"
                if error_output:
                    error_msg += f"Error output:\n{error_output}"
                else:
                    error_msg += f"Exit code: {exit_code}"
                QMessageBox.critical(self, "Launch Error", error_msg)
                self.status_label.setText(f"Error: {app_name} failed to start")
                self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
    
    def _handle_process_error(self, process, app_name, error):
        """Handle QProcess error signals."""
        error_messages = {
            QProcess.ProcessError.FailedToStart: "Failed to start",
            QProcess.ProcessError.Crashed: "Process crashed",
            QProcess.ProcessError.Timedout: "Process timed out",
            QProcess.ProcessError.WriteError: "Write error",
            QProcess.ProcessError.ReadError: "Read error",
            QProcess.ProcessError.UnknownError: "Unknown error"
        }
        error_msg = f"{app_name} error: {error_messages.get(error, 'Unknown error')}"
        QMessageBox.critical(self, "Launch Error", error_msg)
        self.status_label.setText(f"Error: {app_name} failed")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
    
    def _handle_process_finished(self, process, app_name, exit_code, exit_status):
        """Handle QProcess finished signals."""
        if exit_code != 0 and exit_status == QProcess.ExitStatus.CrashExit:
            error_output = process.readAllStandardError().data().decode('utf-8', errors='ignore')
            error_msg = f"{app_name} crashed.\n\n"
            if error_output:
                error_msg += f"Error output:\n{error_output}"
            else:
                error_msg += f"Exit code: {exit_code}"
            QMessageBox.critical(self, "Process Error", error_msg)
            self.status_label.setText(f"Error: {app_name} crashed")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up child processes if needed
        if self.bot_process and self.bot_process.state() == QProcess.ProcessState.Running:
            self.bot_process.terminate()
            self.bot_process.waitForFinished(1000)
        if self.gpu_process and self.gpu_process.state() == QProcess.ProcessState.Running:
            self.gpu_process.terminate()
            self.gpu_process.waitForFinished(1000)
        event.accept()


def main():
    """Main entry point for the Home Launcher application."""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Home Launcher")
        
        # Set application icon if available
        repo_root = Path(__file__).parent
        icon_paths = [
            repo_root / "assets" / "icon.png",
            repo_root / "resources" / "app.ico",
            repo_root / "assets" / "app.ico",
        ]
        for icon_path in icon_paths:
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))
                break
        
        window = HomeWindow()
        # Also set window icon directly
        for icon_path in icon_paths:
            if icon_path.exists():
                window.setWindowIcon(QIcon(str(icon_path)))
                break
        
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error starting Home Launcher: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

