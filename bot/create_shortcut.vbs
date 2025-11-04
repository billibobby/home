' VBScript to create a desktop shortcut for AI Trading Bot GUI
Set WshShell = CreateObject("WScript.Shell")
Set oShellLink = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop") & "\AI Trading Bot.lnk")

' Get the current directory (where this script is located)
strScriptPath = Replace(WScript.ScriptFullName, WScript.ScriptName, "")

' Set shortcut properties
oShellLink.TargetPath = strScriptPath & "launch_gui.bat"
oShellLink.WorkingDirectory = strScriptPath
oShellLink.Description = "AI Trading Bot - XGBoost ML Trading System"
oShellLink.WindowStyle = 1

' Try to set an icon (Python icon)
oShellLink.IconLocation = "C:\Windows\py.exe,0"

oShellLink.Save

' Show success message
MsgBox "Desktop shortcut created successfully!" & vbCrLf & vbCrLf & "You can now launch the AI Trading Bot from your desktop.", vbInformation, "Shortcut Created"

