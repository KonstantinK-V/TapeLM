# Stage 254 continual understanding (shared upper)

**CONTINUAL_UNDERSTAND_NO** domains=wiki->stories->med->news budget=150000 tok/domain

- exam: 0.850 -> 0.825 -> 0.850 -> 0.800 -> 0.800
- max forget vs P1 (hold CE): +0.432 | vs first phase: +0.322 | min mem @bank 22: 1.000 | max leak: 0.400
- cross-domain 2-hop: 4way=1.000 strict=0.667 n=6

| after \ domain | wiki | stories | med | news |
|---|---|---|---|---|
| wiki | ce 4.45 / mem 1.00 | - | - | - |
| stories | ce 5.07 / mem 1.00 | ce 3.48 / mem 1.00 | - | - |
| med | ce 4.70 / mem 1.00 | ce 3.69 / mem 1.00 | ce 4.34 / mem 1.00 | - |
| news | ce 4.14 / mem 1.00 | ce 3.80 / mem 1.00 | ce 4.47 / mem 1.00 | ce 4.43 / mem 1.00 |
