# Stage 254 continual understanding (shared upper)

**CONTINUAL_UNDERSTAND_NO** domains=wiki->stories->med->news budget=4000000 tok/domain

- exam: 0.825 -> 0.800 -> 0.792 -> 0.783 -> 0.833
- max forget (hold CE): +0.471 | min mem @bank 57: 0.923 | max leak: 0.462
- cross-domain 2-hop: 4way=0.444 strict=0.000 n=18

| after \ domain | wiki | stories | med | news |
|---|---|---|---|---|
| wiki | ce 4.31 / mem 1.00 | - | - | - |
| stories | ce 4.70 / mem 1.00 | ce 2.67 / mem 1.00 | - | - |
| med | ce 4.25 / mem 0.92 | ce 3.02 / mem 0.92 | ce 4.00 / mem 1.00 | - |
| news | ce 4.11 / mem 0.92 | ce 3.14 / mem 0.92 | ce 4.00 / mem 0.94 | ce 4.06 / mem 1.00 |
