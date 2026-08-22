"""Re-score stage255 decision JSON with fixed-seed bank metrics (no GPU retrain).

  python _stage255_recompute_decision.py [--decision results/stage255_decision.json] [--tape results/stream255/tape.pt]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import _stage24x_lib as L
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from tokenizers import Tokenizer
RECALL_SEED = 255 + 9000

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--decision', type=str, default='results/stage255_decision.json')
    ap.add_argument('--tape', type=str, default='results/stream255/tape.pt')
    ap.add_argument('--state', type=str, default='results/stream255/state.json')
    args = ap.parse_args()
    dec_path = Path(args.decision)
    if not dec_path.exists():
        raise SystemExit(f'missing {dec_path}')
    d = json.loads(dec_path.read_text(encoding='utf-8'))
    st = json.loads(Path(args.state).read_text(encoding='utf-8'))
    probe_facts = st['probe_facts']
    tape_d = torch.load(Path(args.tape), map_location='cpu', weights_only=False)
    values = tape_d['values']
    K = tape_d['K']
    device = torch.device('cpu')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    Vtok = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, Vtok).to(device)
    model = SelfModelXL(n_char, Vtok).to(device)
    model.load_state_dict(torch.load('checkpoints/stage191_p1_curve.pt', map_location=device, weights_only=False)['model'])
    model.eval()
    bank = FpBank(model, stoi, device)
    history = d.get('history', [])
    baseline_hold = dict(d.get('summary', {}).get('baseline_hold_ce') or {})
    hold_path = Path(args.tape).parent / 'holdouts.pt'
    if hold_path.exists() and (not baseline_hold):
        hb = torch.load(hold_path, map_location='cpu', weights_only=False)
        for dom, batches in hb.items():
            baseline_hold[dom] = s252.fixed_hold_ce(model, batches, char_table, pad_id, device)
    if not baseline_hold and history:
        for dom, ce in history[0].get('hold_ce', {}).items():
            baseline_hold.setdefault(dom, ce)
    baseline_exam = d.get('summary', {}).get('baseline_exam')
    if baseline_exam is None:
        items = s251.load_exam_next(120)
        baseline_exam = s251.next_tok_acc(model, char_table, pad_id, items, device)
    all_vals = list(dict.fromkeys([f['value'] for fs in probe_facts.values() for f in fs] + values))
    for row in history:
        bm = {dom: L.tape_recall_metrics(probe_facts[dom], all_vals, bank, K, values, RECALL_SEED) for dom in probe_facts}
        row['probe_bank'] = bm
        if 'probe_recall' in row:
            del row['probe_recall']
    if not history:
        raise SystemExit('empty history')
    last = history[-1]
    doms = list(last['hold_ce'].keys())
    first_ce = {}
    for r in history:
        for dom, v in r['hold_ce'].items():
            first_ce.setdefault(dom, v)
    forget_vs_first = {dom: last['hold_ce'][dom] - first_ce[dom] for dom in doms}
    forget_vs_p1 = {dom: last['hold_ce'][dom] - baseline_hold.get(dom, first_ce[dom]) for dom in doms}
    rec_curve = [(r['tape_slots'], min((m['top1'] for m in r['probe_bank'].values()))) for r in history]
    mrr_curve = [(r['tape_slots'], min((m['mrr'] for m in r['probe_bank'].values()))) for r in history]
    last_top1 = min((v['top1'] for v in last['probe_bank'].values()))
    last_mrr = min((v['mrr'] for v in last['probe_bank'].values()))
    first_top1 = rec_curve[0][1]
    g_stream = len(history) >= 2
    g_no_forget_p1 = max(forget_vs_p1.values()) <= 0.15
    g_peak_regress = max(forget_vs_first.values()) <= 0.15
    g_grow = last['exam_next_tok'] >= baseline_exam - 0.01
    g_recall_top1 = last_top1 >= max(0.04, first_top1 - 0.03)
    g_recall_mrr = last_mrr >= 0.06
    if g_stream and g_no_forget_p1 and g_grow and g_recall_top1:
        overall = 'STREAM_INGEST_OK'
    elif g_stream and g_no_forget_p1 and (g_grow or g_recall_mrr):
        overall = 'STREAM_INGEST_PARTIAL'
    else:
        overall = 'STREAM_INGEST_NO'
    d['overall'] = overall
    d['gates'] = {'G_streamed': g_stream, 'G_no_forget_vs_P1': g_no_forget_p1, 'G_peak_hold_regress': g_peak_regress, 'G_understanding_holds': g_grow, 'G_recall_top1_floor': g_recall_top1, 'G_recall_mrr_floor': g_recall_mrr, 'G_tape_bounded': last.get('tape_mb', 0) < 2000}
    d['summary'] = {**d.get('summary', {}), 'forget_hold_ce_vs_first_chunk': forget_vs_first, 'forget_hold_ce_vs_P1': forget_vs_p1, 'baseline_hold_ce': baseline_hold, 'baseline_exam': baseline_exam, 'recall_top1_vs_bank': rec_curve, 'recall_mrr_vs_bank': mrr_curve, 'recall_final_top1': last_top1, 'recall_final_mrr': last_mrr}
    d['history'] = history
    d['note'] = (d.get('note', '') + ' Recomputed probe_bank with fixed-seed top1/MRR.').strip()
    dec_path.write_text(json.dumps(d, indent=2), encoding='utf-8')
    print(json.dumps({'overall': overall, 'gates': d['gates'], 'recall_final': (last_top1, last_mrr)}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())