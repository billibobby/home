Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
Dim ScriptDir
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = ScriptDir

' Check if virtual environment exists
If Not FSO.FileExists(ScriptDir & "\venv\Scripts\activate.bat") Then
    ' Show error message if venv doesn't exist
    MsgBox "ERROR: Virtual environment not found!" & vbCrLf & vbCrLf & _
           "The virtual environment has not been set up yet." & vbCrLf & _
           "Please follow these steps:" & vbCrLf & vbCrLf & _
           "1. Open Command Prompt in the application directory" & vbCrLf & _
           "2. Run: python -m venv venv" & vbCrLf & _
           "3. Run: venv\Scripts\activate" & vbCrLf & _
           "4. Run: pip install -r requirements.txt" & vbCrLf & vbCrLf & _
           "Or run launch_novita_gpu_manager.bat for detailed instructions.", _
           vbCritical, "Novita.ai GPU Manager - Setup Required"
    WScript.Quit 1
End If

' Launch application silently
WshShell.Run "cmd /c call venv\Scripts\activate.bat && python main.py", 0, False
Set WshShell = Nothing
Set FSO = Nothing

