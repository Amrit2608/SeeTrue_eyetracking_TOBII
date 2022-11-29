Gui, Add, Button,, Go
Gui, Show, W200 H100

Return

ButtonGo:
if (WinExist("OBS")) {
    ControlClick, Qt5152QWindowIcon8,,,L
    Sleep 5000
    ControlClick, Qt5152QWindowIcon9,,,L
}