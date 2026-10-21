param([int]$pidv, [string]$tag)
$e='D:\PS5_GAMES\NHL27\issue_evidence'; $log="$e\${tag}_watch.log"; Set-Content $log ''
for ($i=0; $i -lt 400; $i++) {
  $proc = Get-Process -Id $pidv -ErrorAction SilentlyContinue
  if (-not $proc) { Add-Content $log 'PROCESS GONE'; break }
  $t = $proc.MainWindowTitle -replace '.*frame: ',''
  $err = Get-Content "$e\${tag}_err.txt" -ErrorAction SilentlyContinue | Select-String -Pattern 'vmaCreateImage failed|out of device memory|placed in system memory'
  $f = ($err | Where-Object { $_.Line -match 'vmaCreateImage failed' }).Count; $v = ($err | Where-Object { $_.Line -match "out of device memory" }).Count; $h = ($err | Where-Object { $_.Line -match "placed in system" }).Count
  $c = Get-Counter -Counter "\GPU Process Memory(pid_$pidv*)\Dedicated Usage","\GPU Process Memory(pid_$pidv*)\Shared Usage","\GPU Process Memory(pid_$pidv*)\Total Committed" -ErrorAction SilentlyContinue
  $ded = 0; $sh = 0; $tc = 0
  foreach ($x in $c.CounterSamples) { if ($x.Path -match 'dedicated') { $ded += $x.CookedValue } elseif ($x.Path -match 'shared') { $sh += $x.CookedValue } else { $tc += $x.CookedValue } }
  Add-Content $log ('{0} fails={1} evictions={2} hostimgs={7} frame={3} dedicatedMiB={4:N0} sharedMiB={5:N0} committedMiB={6:N0}' -f (Get-Date -Format HH:mm:ss), $f, $v, $t, ($ded/1MB), ($sh/1MB), ($tc/1MB), $h)
  Start-Sleep 12
}
