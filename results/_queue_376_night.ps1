# ~5h night queue: static checks + audit battery 371-376 + 377 copy-lane (if wired).
# Log: results\_queue_376_night.out
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File results\_queue_376_night.ps1
#
# 377 GATE (declared before run): reachable up vs 365conn; hit/reach not down >0.02;
# PICK not beaten by COUNT pairwise. D swept: 1 / 4 / 16 (376 audit best D=4 on en).

$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_376_night.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
function Run([string]$label, [scriptblock]$cmd) {
  L "==== $label ===="
  & $cmd
  if ($LASTEXITCODE -ne 0) { L "FAIL exit $LASTEXITCODE $label" } else { L "OK $label" }
}
Set-Content -Path $log -Value "" -Encoding utf8
L "376 night: checks + audits 371-376 + 377copy vs 365conn (4 seeds, D sweep)"

# --- PHASE 1: static wiring (~15 min) ---
Run "check301" { python _check301_wiring.py }
Run "check365" { python _check365_connect.py }
Run "check372rel" { python _check372_relation.py }
Run "check372compass" { python _check372_compass.py }
Run "check337" { python _check337_rank.py }
Run "check376" { python _check376_copy.py }
Run "tape_frames" { python -c "import _tape_frames as t; t.frame_keep(['a b c d e f g h i j k l m n']*20,3,1); print('tape_frames OK')" }

# --- PHASE 2: torch-free audits (~60-90 min) ---
Run "371 family w400" {
  python _audit371_family.py
  Copy-Item results/_stage371_family.json results/_stage371_family_w400.json -Force
}
Run "373 ngram3 w400" {
  python _audit373_pieces.py
  Copy-Item results/_stage373_pieces.json results/_queue373_n3_w400.json -Force
}
Run "373 ngram4 w400" {
  python _audit373_pieces.py --ngram 4
  Copy-Item results/_stage373_pieces.json results/_queue373_n4_w400.json -Force
}
Run "373 ngram5 w1600" {
  python _audit373_pieces.py --ngram 5 --window-lines 1600
  Copy-Item results/_stage373_pieces.json results/_queue373_n5_w1600.json -Force
}
Run "374 en w400" {
  python _audit374_shape.py
  Copy-Item results/_stage374_shape.json results/_queue374_en_w400.json -Force
}
Run "374 en w1600" {
  python _audit374_shape.py --window-lines 1600
  Copy-Item results/_stage374_shape.json results/_queue374_en_w1600.json -Force
}
Run "374 de w8000" {
  python _audit374_shape.py --corpus data/_morph_de.txt --window-lines 8000
  Copy-Item results/_stage374_shape.json results/_queue374_de_w8000.json -Force
}
Run "374 fi e128 w12000" {
  python _audit374_shape.py --corpus data/_morph_fi.txt --window-lines 12000 --endings 128 --top-ends 8
  Copy-Item results/_stage374_shape.json results/_queue374_fi_w12000.json -Force
}
Run "375 en w400 f32" {
  python _audit375_addr.py
  Copy-Item results/_stage375_addr.json results/_queue375_en_w400_f32.json -Force
}
Run "375 en w400 f16" {
  python _audit375_addr.py --func 16
  Copy-Item results/_stage375_addr.json results/_queue375_en_w400_f16.json -Force
}
Run "375 en w400 f64" {
  python _audit375_addr.py --func 64
  Copy-Item results/_stage375_addr.json results/_queue375_en_w400_f64.json -Force
}
Run "375 de w8000" {
  python _audit375_addr.py --corpus data/_morph_de.txt --window-lines 8000
  Copy-Item results/_stage375_addr.json results/_queue375_de_w8000.json -Force
}
Run "376 en w400" {
  python _audit376_copy.py
  Copy-Item results/_stage376_copy.json results/_queue376_en_w400.json -Force
}
Run "376 en w1600" {
  python _audit376_copy.py --window-lines 1600
  Copy-Item results/_stage376_copy.json results/_queue376_en_w1600.json -Force
}
Run "376 de w8000" {
  python _audit376_copy.py --corpus data/_morph_de.txt --window-lines 8000
  Copy-Item results/_stage376_copy.json results/_queue376_de_w8000.json -Force
}
if (Test-Path data/_morph_fi.txt) {
  Run "376 fi w12000" {
    python _audit376_copy.py --corpus data/_morph_fi.txt --window-lines 12000
    Copy-Item results/_stage376_copy.json results/_queue376_fi_w12000.json -Force
  }
}
Run "373 de w8000 n4" {
  python _audit373_pieces.py --corpus data/_morph_de.txt --window-lines 8000 --ngram 4
  Copy-Item results/_stage373_pieces.json results/_queue373_de_w8000_n4.json -Force
}
Run "373 fi w12000 n4" {
  python _audit373_pieces.py --corpus data/_morph_fi.txt --window-lines 12000 --ngram 4
  Copy-Item results/_stage373_pieces.json results/_queue373_fi_w12000_n4.json -Force
}

# --- read pooled controls (no train) ---
L "==== _read299 existing 365conn ===="
$f = @(Get-ChildItem out\_stage289_decision_365conn_s*.json,results\stage289_decision_365conn_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_365conn_night.txt }

# --- PHASE 3: stage 377 copy-lane vs 365conn (~3-4h if --copy-lane wired) ---
$help = python _stage289_derivation.py --help 2>&1 | Out-String
$hasCopy = $help -match 'copy-lane|copy_lane|copy-depth'
if (-not $hasCopy) {
  L "SKIP 377 stage: --copy-lane not in _stage289_derivation.py yet; audits above are the night run"
  L "When wired: re-run this script or only PHASE 3 block"
} else {
  $stand = @(
    '--tape','frames','--frame-max','3','--min-mentions','1',
    '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
    '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
    '--tape-sample','region',
    '--objective','reward','--addresses','1500','--reach-max-q','8000',
    '--import-k','1','--gamma','1.0',
    '--probe-period','1000','--probe-size','60','--cpu',
    '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
    '--min-fillers','1','--connect','--copy-lane'
  )
  $seeds = @(1337, 8642, 2890, 4711)
  $wiki = "data\_wikitext103_train.txt"
  foreach ($d in @(1, 4, 16)) {
    foreach ($s in $seeds) {
      $tag = "377copy_d${d}_s$s"
      $out = "out/_stage289_decision_$tag.json"
      L "==== $tag ===="
      python -u _stage289_derivation.py @stand --copy-depth $d --wiki $wiki --seed $s --run-tag $tag --out $out
      if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
    }
  }
  L "==== _read299 377copy d4 (primary arm) ===="
  $f = @(Get-ChildItem out\_stage289_decision_377copy_d4_s*.json,results\stage289_decision_377copy_d4_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
  if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_377copy_d4.txt }
  L "==== _read299 365conn control ===="
  $f = @(Get-ChildItem out\_stage289_decision_365conn_s*.json,results\stage289_decision_365conn_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
  if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_365conn_ctrl377.txt }
}

L "DONE 376 night queue"
