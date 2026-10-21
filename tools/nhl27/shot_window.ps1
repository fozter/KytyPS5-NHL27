param([string]$Name = "win", [string]$InFile = "")
# Crops the kyty_emulator window region out of a full-screen capture (or a fresh capture).
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System; using System.Runtime.InteropServices;
public class WR { [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f); }
'@
$p = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $p) { Write-Output "no emulator window"; exit 1 }
$h = $p.MainWindowHandle
# raise above everything without activating (flags 0x0053 = SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE|SWP_SHOWWINDOW)
[WR]::SetWindowPos($h, [IntPtr](-1), 0, 0, 0, 0, 0x0053) | Out-Null
Start-Sleep -Milliseconds 250
$r = New-Object WR+RECT; [WR]::GetWindowRect($h, [ref]$r) | Out-Null
$w = $r.R - $r.L; $h = $r.B - $r.T
$b = New-Object System.Drawing.Bitmap($w, $h)
$g = [System.Drawing.Graphics]::FromImage($b)
$g.CopyFromScreen($r.L, $r.T, 0, 0, $b.Size); $g.Dispose()
$file = "D:\PS5_GAMES\NHL27\screenshots\$Name`_$(Get-Date -Format HHmmss).png"
$b.Save($file, [System.Drawing.Imaging.ImageFormat]::Png); $b.Dispose()
Write-Output "shot: $file ($w x $h)"
