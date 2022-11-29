#SingleInstance, force

Gui, Add, Button,, Go
Gui, Show, W200 H100

SetTitleMatchMode, 2

NumPictures := 5
DurationPerPicture = 10000

Return

ButtonGo:
obs_win := WinExist("OBS")
photos_win := WinExist("Photos")
if (obs_win > 0 and photos_win > 0) 
{
    Gui, Minimize

    ; Start Recording
    WinActivate, ahk_id %obs_win%
    Send, ^+r
    WinSet, Bottom,, ahk_id %obs_win%

    ; Show pictures
    WinActivate, ahk_id %photos_win%
    Sleep %DurationPerPicture%
    Loop % NumPictures - 1 
    {
        Send, {Right}
        Sleep %DurationPerPicture%
    }

    ; Stop Recording
    WinActivate, ahk_id %obs_win%
    Send, ^+r
    WinSet, Bottom,, ahk_id %obs_win%

    ; Back to start of pictures
    WinActivate, ahk_id %photos_win%
    Loop % NumPictures - 1 
    {
        Send, {Left}
        Sleep 20 ;SLeep briefly to ensure that all arrow sends are processed
    }

    Gui, Restore
}