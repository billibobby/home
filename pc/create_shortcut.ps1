# Create desktop shortcut for Novita GPU Manager
$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $Desktop "Novita GPU Manager.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = Join-Path $PSScriptRoot "launch_novita_gpu_manager_silent.vbs"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "Novita.ai GPU Instance Manager"
$Shortcut.Save()
Write-Host "Desktop shortcut created at: $ShortcutPath"

