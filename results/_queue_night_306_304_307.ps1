$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly

function Wait-StageFree {
  while ($true) {
    $busy = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -and
        ($_.CommandLine -like "*_stage289_derivation.py*")
      }
    if (-not $busy) { return }
    Write-Host ("==== wait stage still running pid={0} ====" -f ($busy.ProcessId -join ","))
    Start-Sleep -Seconds 60
  }
}

function Run-B {
  param(
    [string[]]$Extra,
    [int]$Seed,
    [string]$Tag
  )
  Write-Host ("==== {0} seed {1} ====" -f $Tag, $Seed)
  $args = @(
    "-u", "_stage289_derivation.py",
    "--tape", "frames", "--frame-max", "3", "--min-mentions", "1",
    "--fp", "hash", "--write-fp", "hash", "--ink", "mean", "--write-ink", "mean", "--words", "ascii",
    "--reach", "--reach-no-refuse", "--reach-lookahead",
    "--frame-fp", "fillers", "--tape-sample", "region",
    "--objective", "reward", "--addresses", "1500", "--reach-max-q", "2000",
    "--probe-period", "1000", "--probe-size", "60", "--cpu",
    "--train-steps", "4000", "--seed", "$Seed", "--run-tag", $Tag
  ) + $Extra
  & python @args
  if ($LASTEXITCODE -ne 0) { throw "failed $Tag exit $LASTEXITCODE" }
}

function Read-Group {
  param([string]$Glob)
  Write-Host "==== _read299 $Glob ===="
  $files = @(Get-ChildItem $Glob -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
  if ($files.Count -eq 0) { Write-Host "no files for $Glob"; return }
  python _read299.py @files --held
}

Wait-StageFree

Write-Host "==== preflight _check301_wiring ===="
python _check301_wiring.py
if ($LASTEXITCODE -ne 0) { throw "wiring failed" }

$seeds = @(1337, 8642, 5200, 2718)

foreach ($s in $seeds) {
  Run-B -Extra @("--reach-confirm") -Seed $s -Tag "306_conf_s$s"
}
Read-Group "results\stage289_decision_306_conf_s*.json"

foreach ($s in $seeds) {
  Run-B -Extra @("--reach-line") -Seed $s -Tag "304_line_s$s"
}
Read-Group "results\stage289_decision_304_line_s*.json"

foreach ($s in $seeds) {
  Run-B -Extra @("--flat") -Seed $s -Tag "307_flat_s$s"
}
Read-Group "results\stage289_decision_307_flat_s*.json"

Write-Host "==== night queue 306/304/307 done ===="
