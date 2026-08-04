# Stage 254 continual understanding (operators-only)
**CONTINUAL_UNDERSTAND_OK** domains=wiki->stories->med->news budget=150000 tok/domain

- exam: 0.850 -> 0.850 -> 0.850 -> 0.850 -> 0.850
- max forget vs P1 (hold CE): +0.000 | vs first phase: +0.000 | min mem @bank 22: 1.000 | max leak: 0.300
- cross-domain 2-hop: 4way=1.000 strict=0.667 n=6

| after \ domain | wiki | stories | med | news |
|---|---|---|---|---|
| wiki | ce 4.37 / mem 1.00 | - | - | - |
| stories | ce 4.37 / mem 1.00 | ce 6.36 / mem 1.00 | - | - |
| med | ce 4.37 / mem 1.00 | ce 6.36 / mem 1.00 | ce 4.03 / mem 1.00 | - |
| news | ce 4.37 / mem 1.00 | ce 6.36 / mem 1.00 | ce 4.03 / mem 1.00 | ce 4.33 / mem 1.00 |
