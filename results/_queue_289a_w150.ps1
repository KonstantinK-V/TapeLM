$ErrorActionPreference = "Continue"
$log = "results\_queue_289a_w150.out"
"===== 289a w150 fast-grouping (6) $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8
cmd /c "python -u _stage289a_presupposition.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 > `"results\_stage289a_full_w150.out`" 2>&1"
"EXIT_FULL:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
"QUEUE_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
