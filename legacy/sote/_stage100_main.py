def _subsample(lines, n, seed):
    if not lines or n <= 0 or len(lines) <= n:
        return list(lines)
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    return [lines[i] for i in idx[:n]]


def main():
    cfg = Config()
    # Stage100 knobs
    cfg.c100_max_lines = getattr(cfg, "c100_max_lines", 100000)
    cfg.c100_ft_steps = getattr(cfg, "c100_ft_steps", 50000)
    cfg.c100_ft_lr = getattr(cfg, "c100_ft_lr", 1e-3)
    cfg.c100_eval_every = getattr(cfg, "c100_eval_every", 1000)
    cfg.c100_batch = getattr(cfg, "c100_batch", 8)
    cfg.c100_warmup = getattr(cfg, "c100_warmup", 200)
    cfg.c100_fat_frac = getattr(cfg, "c100_fat_frac", 0.75)
    cfg.c100_eval_seen_n = getattr(cfg, "c100_eval_seen_n", 600)
    cfg.c100_eval_story_n = getattr(cfg, "c100_eval_story_n", 400)
    cfg.c100_eval_rare_n = getattr(cfg, "c100_eval_rare_n", 120)
    cfg.c100_d_model = getattr(cfg, "c100_d_model", 256)
    cfg.c100_n_layers = getattr(cfg, "c100_n_layers", 2)
    cfg.c100_n_heads = getattr(cfg, "c100_n_heads", 4)
    cfg.c100_max_len = getattr(cfg, "c100_max_len", 16)
    cfg.c100_dropout = getattr(cfg, "c100_dropout", 0.1)
    cfg.c100_obj_partial = getattr(cfg, "c100_obj_partial", 0.25)
    cfg.c100_story_lift_pp = getattr(cfg, "c100_story_lift_pp", 0.08)
    cfg.c100_n_fat = getattr(cfg, "c100_n_fat", 200)
    cfg.c100_fat_copies = getattr(cfg, "c100_fat_copies", 40)
    cfg.c100_story_keep_frac = getattr(cfg, "c100_story_keep_frac", 0.50)
    cfg.c100_hold_frac = getattr(cfg, "c100_hold_frac", 0.15)
    # feed into build_ts_repeat_mix (uses c87_*)
    cfg.c87_n_fat = int(cfg.c100_n_fat)
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = int(cfg.c100_fat_copies)
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = float(cfg.c100_story_keep_frac)
    cfg.c87_hold_frac = float(cfg.c100_hold_frac)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Device:", device, f"({torch.cuda.get_device_name(device)})")
    else:
        print("Device: cpu")

    print("Stage 100: scale atom ~100k TinyStories (recipe98; 6h budget)")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    assert parent85.exists(), f"missing {parent85}"
    # load_foundation_85 uses PARENT85 global — patch via load path arg
    stack, f85 = load_foundation_85(device, cfg, path=parent85)
    print(f"  foundation {parent85.name} dim={f85.get('dim')}", flush=True)

    raw100 = ROOT / "data" / "_tinystories_raw_100k.txt"
    raw_fallback = ROOT / "data" / "_tinystories_raw_chunk.txt"
    corpus_path = ROOT / "data" / "external_tinystories_100k_85.txt"

    if not corpus_path.exists():
        raw = raw100 if raw100.exists() and raw100.stat().st_size > 1_000_000 else raw_fallback
        assert raw.exists(), f"need raw TinyStories at {raw100} or {raw_fallback}"
        print(f"  filtering {raw.name} -> {corpus_path.name} max_lines={cfg.c100_max_lines} ...", flush=True)
        phrases, meta_src = filter_tinystories_chunk(
            raw, corpus_path,
            max_lines=int(cfg.c100_max_lines),
            max_word_len=int(cfg.max_word_len),
            seed=272,
        )
        print(f"  filtered n={len(phrases)} meta={meta_src}", flush=True)
    else:
        phrases = [
            ln.strip() for ln in corpus_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        meta_src = {"source": corpus_path.name, "n": len(phrases), "reused": True}
        print(f"  reused corpus n={len(phrases)}", flush=True)

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    print(f"  mix meta={ {k: meta[k] for k in meta if k != 'top_triple_freq'} }", flush=True)

    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    print(f"  building fps V={len(all_words)} ...", flush=True)
    word_fps = F.normalize(torch.stack([stack.w(w).detach() for w in all_words], 0).to(device), dim=-1)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}

    # eval subsets (speed)
    ev_seen = _subsample(hold_seen, int(cfg.c100_eval_seen_n), 1001)
    ev_rare = _subsample(hold_rare, int(cfg.c100_eval_rare_n), 1002)
    ev_story = _subsample(hold_story, int(cfg.c100_eval_story_n), 1003)
    # final eval slightly larger
    fin_seen = _subsample(hold_seen, min(len(hold_seen), max(1200, int(cfg.c100_eval_seen_n))), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), max(120, int(cfg.c100_eval_rare_n))), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), max(800, int(cfg.c100_eval_story_n))), 2003)

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    fat_pairs = lines_to_pairs(fat_lines, stoi) if fat_lines else []
    story_pairs = lines_to_pairs(story_lines, stoi) if story_lines else lines_to_pairs(train, stoi)
    if not fat_pairs:
        fat_pairs = story_pairs
    print(
        f"  train fat_pairs={len(fat_pairs)} story_pairs={len(story_pairs)} "
        f"eval_seen={len(ev_seen)} eval_story={len(ev_story)}",
        flush=True,
    )

    model = WordIdTransformer(
        n_vocab=len(surf),
        d_model=int(cfg.c100_d_model),
        n_heads=int(cfg.c100_n_heads),
        n_layers=int(cfg.c100_n_layers),
        max_len=int(cfg.c100_max_len),
        dropout=float(cfg.c100_dropout),
    ).to(device)
    model.init_from_fps(word_fps)
    n_params = sum(p.numel() for p in model.parameters())

    batch_sz = int(cfg.c100_batch)
    steps = int(cfg.c100_ft_steps)
    peak_lr = float(cfg.c100_ft_lr)
    warmup = int(cfg.c100_warmup)
    fat_frac = float(cfg.c100_fat_frac)
    every = int(cfg.c100_eval_every)
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    rng = random.Random(cfg.seed)

    print(
        f"  model params={n_params/1e6:.2f}M batch={batch_sz} steps={steps} "
        f"warmup={warmup} lr={peak_lr} fat_frac={fat_frac} eval_every={every}",
        flush=True,
    )

    def _set_lr(step: int) -> float:
        lr = _warmup_then_constant(step, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    def _sample_batch():
        n_fat = max(1, int(round(batch_sz * fat_frac))) if fat_pairs else 0
        n_fat = min(n_fat, batch_sz)
        n_st = batch_sz - n_fat
        ex = [rng.choice(fat_pairs) for _ in range(n_fat)]
        ex += [rng.choice(story_pairs) for _ in range(n_st)]
        rng.shuffle(ex)
        return ex

    def _eval_hold(lines):
        return eval_id_capacity_suite(model, lines, train, surf, stoi, device) if lines else None

    best = {
        "step": 0, "obj": 0.0, "rel": 0.0, "story_all": 0.0,
        "seen": None, "rare": None, "story": None,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def _snap(step: int):
        model.eval()
        seen = _eval_hold(ev_seen)
        rare = _eval_hold(ev_rare)
        story = _eval_hold(ev_story)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({
            "step": step, "obj_seen": obj_s, "rel_seen": rel_s,
            "story_all": st_all, "obj_rare": rare["obj"]["hit1"],
            "obj_story": story["obj"]["hit1"],
            "head_seen": seen["freq"].get("head21p", {}).get("model", 0.0),
        })
        # prefer story lift while keeping joint SEEN
        key = (1 if rel_s >= 0.50 else 0, st_all, obj_s, rel_s)
        bkey = (
            1 if best["rel"] >= 0.50 else 0,
            best["story_all"], best["obj"], best["rel"],
        )
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, rel=rel_s, story_all=st_all,
                seen=seen, rare=rare, story=story,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        print(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}% obj={story['obj']['hit1']*100:.1f}% | "
            f"RARE obj={rare['obj']['hit1']*100:.1f}%",
            flush=True,
        )
        model.train()

    print("\n=== FT scale ===", flush=True)
    _snap(0)
    model.train()
    for step in range(1, steps + 1):
        _set_lr(step)
        packed = collate_word_id_batch(_sample_batch(), stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, pad_mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            _snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    print("\n=== Final eval (larger subsample) ===", flush=True)
    f_seen = _eval_hold(fin_seen)
    f_rare = _eval_hold(fin_rare)
    f_story = _eval_hold(fin_story)
    obj_s = f_seen["obj"]["hit1"]
    rel_s = f_seen["roles"].get("rel", {}).get("hit1", 0.0)
    all_s = f_seen["roles"].get("ALL", {}).get("hit1", 0.0)
    head_s = f_seen["freq"].get("head21p", {}).get("model", 0.0)
    st_all = f_story["roles"].get("ALL", {}).get("hit1", 0.0)
    st_obj = f_story["obj"]["hit1"]
    rare_obj = f_rare["obj"]["hit1"]

    ref98 = {"obj": 0.413, "rel": 0.960, "story_all": 0.092, "rare_obj": 0.133}
    lift_story = st_all - ref98["story_all"]

    if (
        obj_s >= float(cfg.c100_obj_partial)
        and rel_s >= 0.50
        and lift_story >= float(cfg.c100_story_lift_pp)
    ):
        verdict = "PASS"
    elif obj_s >= float(cfg.c100_obj_partial) and rel_s >= 0.50 and lift_story >= 0.03:
        verdict = "PARTIAL"
    elif obj_s + 0.02 >= ref98["obj"] and rel_s >= 0.50 and abs(lift_story) < 0.03:
        verdict = "CEILING"
    elif obj_s >= float(cfg.c100_obj_partial) and rel_s >= 0.50:
        verdict = "PARITY_ATOM"
    else:
        verdict = "FAIL"

    lines = [
        "SOTE Stage 100 — scale atom ~100k TinyStories (recipe98 hparams)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"device: {device}",
        f"foundation: {parent85.name} dim={cfg.dim}",
        f"corpus: {corpus_path.name} n_phrases={len(phrases)} meta={meta_src}",
        f"mix: { {k: meta[k] for k in meta if k != 'top_triple_freq'} }",
        f"V={len(surf)} params={n_params}",
        f"train: batch={batch_sz} steps={steps} warmup={warmup} lr={peak_lr} "
        f"fat_frac={fat_frac} emb=x1 (Stage98 recipe)",
        "NO overwrite of F85 dual-channel FROZEN. NO soft@5.",
        "",
        "=== Final ===",
        f"  best_step(during)={best['step']}",
        f"  SEEN  obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% ALL={all_s*100:.1f}% head={head_s*100:.1f}%",
        f"  RARE  obj={rare_obj*100:.1f}%",
        f"  STORY ALL={st_all*100:.1f}% obj={st_obj*100:.1f}% "
        f"rel={f_story['roles'].get('rel',{}).get('hit1',0)*100:.1f}%",
        "",
        "=== vs Stage98 mini ===",
        f"  SEEN obj {obj_s*100:.1f}% vs {ref98['obj']*100:.1f}%",
        f"  SEEN rel {rel_s*100:.1f}% vs {ref98['rel']*100:.1f}%",
        f"  STORY ALL {st_all*100:.1f}% vs {ref98['story_all']*100:.1f}% (lift {lift_story*100:+.1f}pp)",
        f"  RARE obj {rare_obj*100:.1f}% vs {ref98['rare_obj']*100:.1f}%",
        "",
        "=== Adequacy ===",
        "  Ask: was Stage98 STORY soft-ceiling from mini data?",
        f"  PASS if STORY ALL lift >= {cfg.c100_story_lift_pp*100:.0f}pp with SEEN held.",
        "  CEILING if SEEN held and STORY flat (|lift|<3pp).",
        "",
        f"=== Verdict: {verdict} ===",
    ]
    # freq rows
    lines.append("  SEEN freq model/bigram:")
    for k in ("head21p", "mid6_20", "low2_5", "rare1"):
        if k not in f_seen["freq"]:
            continue
        v = f_seen["freq"][k]
        lines.append(
            f"    {k:8s} n={v['n']:5d}  model={v['model']*100:5.1f}%  bigram={v['bigram']*100:5.1f}%"
        )
    report = "\n".join(lines) + "\n"
    print()
    print(report)
    OUT_TXT.write_text(report, encoding="utf-8")
    (RES / f"stage100_scale_100k_{verdict}.txt").write_text(report, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "stage": 100,
        "verdict": verdict,
        "best_step": best["step"],
        "n_phrases": len(phrases),
        "V": len(surf),
        "meta": {k: meta[k] for k in meta if k != "top_triple_freq"},
        "meta_src": meta_src,
        "seen": {"obj": obj_s, "rel": rel_s, "all": all_s, "head": head_s, "freq": f_seen["freq"]},
        "rare": {"obj": rare_obj},
        "story": {"all": st_all, "obj": st_obj},
        "lift_story_pp": lift_story,
        "ref98": ref98,
        "curve": curve,
        "cfg": {
            "c100_max_lines": cfg.c100_max_lines,
            "c100_ft_steps": cfg.c100_ft_steps,
            "c100_batch": cfg.c100_batch,
            "c100_ft_lr": cfg.c100_ft_lr,
            "c100_warmup": cfg.c100_warmup,
            "c100_fat_frac": cfg.c100_fat_frac,
        },
    }, indent=2, default=str), encoding="utf-8")
    torch.save({
        "stage": 100,
        "verdict": verdict,
        "word_tf": best["state"],
        "surfaces": surf,
        "word_fps": word_fps.detach().cpu(),
        "cfg": asdict(cfg),
        "meta": {k: meta[k] for k in meta if k != "top_triple_freq"},
        "foundation": str(parent85),
        "corpus": str(corpus_path),
    }, OUT_CKPT)
    print(f"Saved: {OUT_CKPT}")


if __name__ == "__main__":
    main()
