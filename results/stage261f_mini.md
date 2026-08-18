# Stage 261f word votes (zero-train)

**WORD_VOTES_BEATS_MEAN** candidates=4338 vocab=11195 soft=False typo=0.0

- 20-way **0.432** (chance 0.050, popularity floor 0.021) vs 261 ctx_fp mean **0.226**
- open top1 **0.246** vs popularity floor **0.000** (261 mean: 0.034), mrr 0.282, median rank 76
- by overlap: low **0.024** vs high **0.467**
- **silence:** tie_at_zero **0.488** (low-ov **0.864** / high **0.112**); low-ov miss is silence **0.885**; top1 low-ov | gold>0 **0.174**
- trained parameters: **0**
