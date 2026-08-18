$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_341_speak.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "341 speak-batch: matched question budget - ctrl N=4000 vs speak B=8 steps=500 weight=1.0"

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu'
)
$seeds = @(1337, 8642, 2890, 4711)
$news = "data\_stage254_news.txt"
$N = 4000
$B = 8
$Nspeak = [int]($N / $B)   # 500

# --- ctrl: same recipe as 336/340, no speak term ---
L "==== BLOCK 341ctrl: train-steps $N (no --speak-batch) ===="
foreach ($s in $seeds) {
  $tag = "341ctrl_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $news --seed $s --run-tag $tag `
    --train-steps $N --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$fc = @(Get-ChildItem out\_stage289_decision_341ctrl_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($fc.Count -gt 0) {
  L "==== _read299 341ctrl (held) ===="
  python _read299.py @fc --held | Tee-Object -FilePath results\_read341ctrl_held.txt
}

# --- 341: speak-batch B, steps N/B ---
L "==== BLOCK 341speak: train-steps $Nspeak --speak-batch $B --speak-weight 1.0 ===="
foreach ($s in $seeds) {
  $tag = "341speak_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $news --seed $s --run-tag $tag `
    --train-steps $Nspeak --speak-batch $B --speak-weight 1.0 --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$fs = @(Get-ChildItem out\_stage289_decision_341speak_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($fs.Count -gt 0) {
  L "==== _read299 341speak (held) ===="
  python _read299.py @fs --held | Tee-Object -FilePath results\_read341speak_held.txt
}

L "DONE 341 ctrl+speak (GATE-WO claim vs 0.750/0.663; veto ROUTER + PICK vs COUNT)"
