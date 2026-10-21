param(
    [Parameter(Mandatory=$true)][string[]]$Keys,
    [int]$DelayMs = 900,
    [int]$HoldMs = 400,
    [string]$ShotName = "",
    [switch]$Probe
)
# Game driver: correct Win32 INPUT marshaling (x64: sizeof(INPUT) must be 40).
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class Drive {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSEINPUT { public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }
    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }
    [StructLayout(LayoutKind.Explicit)]
    public struct InputUnion { [FieldOffset(0)] public MOUSEINPUT mi; [FieldOffset(0)] public KEYBDINPUT ki; }
    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT { public uint type; public InputUnion U; }
    [DllImport("user32.dll", SetLastError=true)] public static extern uint SendInput(uint n, INPUT[] i, int size);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int cx, int cy, uint flags);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    public const uint MF_LEFTDOWN = 0x0002, MF_LEFTUP = 0x0004;
    public const uint KF_KEYUP = 0x0002;
    public static IntPtr HWND_NOTOPMOST = new IntPtr(-2);
    public static int SizeOfInput() { return Marshal.SizeOf(typeof(INPUT)); }
    public static uint ClickDown() {
        INPUT[] a = new INPUT[1]; a[0].type = 0; a[0].U.mi.dwFlags = MF_LEFTDOWN;
        return SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
    }
    public static uint ClickUp() {
        INPUT[] a = new INPUT[1]; a[0].type = 0; a[0].U.mi.dwFlags = MF_LEFTUP;
        return SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
    }
    public static uint KeyDown(ushort vk, bool extended) {
        INPUT[] a = new INPUT[1]; a[0].type = 1; a[0].U.ki.wVk = vk;
        if (extended) a[0].U.ki.dwFlags = 0x0001; // KEYEVENTF_EXTENDEDKEY
        return SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
    }
    public static uint KeyUp(ushort vk, bool extended) {
        INPUT[] a = new INPUT[1]; a[0].type = 1; a[0].U.ki.wVk = vk;
        a[0].U.ki.dwFlags = KF_KEYUP | (extended ? 0x0001u : 0u);
        return SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
    }
}
'@

$sz = [Drive]::SizeOfInput()
if ($sz -ne 40) { Write-Output "FATAL: sizeof(INPUT)=$sz (expected 40)"; exit 1 }
Write-Output "sizeof(INPUT)=$sz OK"

$map = @{ up=0x26; down=0x28; left=0x25; right=0x27; j=0x4A; k=0x4B; l=0x4C; i=0x49; enter=0x0D; q=0x51; e=0x45; z=0x5A; c=0x43; w=0x57; a=0x41; s=0x53; d=0x44;
          lstick_up=0x57; lstick_left=0x41; lstick_down=0x53; lstick_right=0x44; cross=0x4A; circle=0x4C; square=0x4B; triangle=0x49;
          f12=0x7B; rdcapture=0x7B }

$p = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $p) { Write-Output "no emulator window"; exit 1 }
$hwnd = $p.MainWindowHandle

# 0a) Restore from minimized (SW_RESTORE = 9) — a minimized window has a 160x28 rect
[Drive]::ShowWindow($hwnd, 9) | Out-Null
Start-Sleep -Milliseconds 300

# 0) Force the window above everything (HWND_TOPMOST = -1, keep size/pos: flags 0x0043
#    = SWP_NOSIZE|SWP_NOMOVE|SWP_SHOWWINDOW) so the click cannot land on a window
#    that covers the emulator.
[Drive]::SetWindowPos($hwnd, [IntPtr](-1), 0, 0, 0, 0, 0x0043) | Out-Null
Start-Sleep -Milliseconds 200

# 1) Clear topmost and restore window size/pos (flags 0x0040 = SWP_SHOWWINDOW)
[Drive]::SetWindowPos($hwnd, [IntPtr](-2), 600, 200, 1280, 720, 0x0040) | Out-Null
Start-Sleep -Milliseconds 300

# 1b) Re-assert foreground programmatically before the click
[Drive]::SetForegroundWindow($hwnd) | Out-Null
Start-Sleep -Milliseconds 150

# 2) Real click on client center -> activates the window
$r = New-Object Drive+RECT
[Drive]::GetWindowRect($hwnd, [ref]$r) | Out-Null
$cx = ($r.L + $r.R) / 2; $cy = ($r.T + $r.B) / 2
[Drive]::SetCursorPos($cx, $cy) | Out-Null
Start-Sleep -Milliseconds 120
$cd = [Drive]::ClickDown()
Start-Sleep -Milliseconds 120
$cu = [Drive]::ClickUp()
Start-Sleep -Milliseconds 400
Write-Output "click down=$cd up=$cu focus_ok: $([Drive]::GetForegroundWindow() -eq $hwnd)"

# 3) Optional mouse-motion probe
if ($Probe) {
    1..6 | ForEach-Object {
        [Drive]::SetCursorPos(($cx - 150 + $_ * 50), $cy) | Out-Null
        Start-Sleep -Milliseconds 130
    }
    Write-Output "probe done"
}

# 4) Keys (real injection now; arrows/etc. are extended keys)
$extendedKeys = @(0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5C)
foreach ($k in $Keys) {
    $vk = $map[$k.ToLower()]
    if (-not $vk) { Write-Output "unknown key: $k"; continue }
    $ext = $extendedKeys -contains $vk
    $kd = [Drive]::KeyDown([uint16]$vk, $ext)
    Start-Sleep -Milliseconds $HoldMs
    $ku = [Drive]::KeyUp([uint16]$vk, $ext)
    Write-Output "pressed: $k (down=$kd up=$ku ext=$ext)"
    Start-Sleep -Milliseconds $DelayMs
}

# 5) Screenshot
if ($ShotName -ne "") {
    Start-Sleep -Seconds 4
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $b = New-Object System.Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height)
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.CopyFromScreen(0, 0, 0, 0, $b.Size)
    $g.Dispose()
    $dir = "D:\PS5_GAMES\NHL27\screenshots"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $file = Join-Path $dir "$ShotName`_$(Get-Date -Format HHmmss).png"
    $b.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $b.Dispose()
    Write-Output "shot: $file"
}
