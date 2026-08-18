$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
New-Item -ItemType Directory -Force -Path out | Out-Null

function Run-309 {
  param([string[]]$Extra, [int]$Seed, [string]$Tag)
  Write-Host "==== $Tag seed $Seed ===="
  $args = @(
    "-u", "_stage289_derivation.py",
    "--tape", "frames", "--frame-max", "3", "--min-mentions", "1",
    "--fp", "hash", "--write-fp", "hash", "--ink", "mean", "--write-ink", "mean", "--words", "ascii",
    "--tape-sample", "region", "--addresses", "1500", "--pair", "--frame-fp", "fillers",
    "--probe-period", "1000", "--probe-size", "60", "--cpu",
    "--train-steps", "4000", "--seed", "$Seed", "--run-tag", $Tag
  ) + $Extra
  & python @args
  if ($LASTEXITCODE -ne 0) { throw "failed $Tag exit $LASTEXITCODE" }
  Copy-Item -Force "results\stage289_decision_$Tag.json" "out\_stage289_decision_$Tag.json"
}

Write-Host "==== preflight ===="
python _check301_wiring.py
if ($LASTEXITCODE -ne 0) { throw "wiring failed" }
python _check309_pair.py
if ($LASTEXITCODE -ne 0) { throw "pair check failed" }

$seeds = @(1337, 8642, 2890, 4711)

foreach ($s in $seeds) {
  Run-309 -Extra @() -Seed $s -Tag "309b_seq_s$s"
}
Write-Host "==== _read309 seq ===="
python _read309.py (Get-ChildItem out\_stage289_decision_309b_seq_s*.json | ForEach-Object { $_.FullName }) --held

foreach ($s in $seeds) {
  Run-309 -Extra @("--pair-independent") -Seed $s -Tag "309b_indep_s$s"
}
Write-Host "==== _read309 indep ===="
python _read309.py (Get-ChildItem out\_stage289_decision_309b_indep_s*.json | ForEach-Object { $_.FullName }) --held
Write-Host "==== 309b seq+indep done ===="
