Add-Type -AssemblyName System.Windows.Forms
$global:shell = New-Object -ComObject WScript.Shell
$shell.Popup("Your Makefile task finished successfully!", 5, "Task Completed !", 64)
