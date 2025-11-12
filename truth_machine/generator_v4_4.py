#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import random
import sys
import time
import argparse
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Iterable, Tuple, Dict, Optional
import curses

# ============================
# Tokenizer / Detokenizer
# ============================

_PUNCT_STICKY_RIGHT = {".", ",", ":", ";", "!", "?", "…", ")", "]", "}", "»", "”", "’"}
_PUNCT_STICKY_LEFT  = {"(", "[", "{", "«", "“", "‘", "€", "$", "£", "¥", "₹"}
_QUOTE = {"'", "’", "“", "”", '"'}

def tokenize(text: str) -> List[str]:
    rough = text.strip().split()
    tokens: List[str] = []
    for tok in rough:
        while tok and tok[0] in _PUNCT_STICKY_LEFT:
            tokens.append(tok[0]); tok = tok[1:]
        right_buf = []
        while tok and tok[-1] in _PUNCT_STICKY_RIGHT:
            right_buf.append(tok[-1]); tok = tok[:-1]
        if tok:
            tokens.append(tok)
        tokens.extend(reversed(right_buf))
    return [t for t in tokens if t]

def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if not out:
            out.append(t)
        elif t in _PUNCT_STICKY_RIGHT or t in _QUOTE:
            out[-1] = out[-1] + t
        elif out[-1] in _PUNCT_STICKY_LEFT or out[-1] in _QUOTE:
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)

# ============================
# Data loading
# ============================

def load_from_vocab_csv(path: str, col: str = "word") -> List[str]:
    words = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if col not in reader.fieldnames:
            print(f"Column '{col}' not found in {path}. Columns: {reader.fieldnames}")
            sys.exit(1)
        for row in reader:
            w = (row.get(col) or "").strip()
            if w:
                words.append(w)
    return words

def load_texts_csv(path: str, text_col: str) -> List[str]:
    texts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            print(f"Column '{text_col}' not found in {path}. Columns: {reader.fieldnames}")
            sys.exit(1)
        for row in reader:
            t = (row.get(text_col) or "").strip()
            if t:
                texts.append(t)
    return texts

def load_wordlist(path: Optional[str]) -> set:
    if not path:
        return set()
    st = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                st.add(w.lower())
    return st

# ---- synthetic docs (anti-loop) ----
def build_synthetic_docs(words: List[str], order: int,
                         passes: int = 32,
                         chunk_min: int = 60,
                         chunk_max: int = 180) -> List[List[str]]:
    if not words:
        return []
    docs: List[List[str]] = []
    for _ in range(max(1, passes)):
        w = words[:]
        random.shuffle(w)
        i = 0
        n = len(w)
        while i < n:
            L = random.randint(chunk_min, chunk_max)
            if L < order + 1:
                L = order + 1
            docs.append(w[i:i+L])
            i += L
    return docs

# ============================
# Markov model
# ============================

class MarkovModel:
    def __init__(self, order: int = 3, temperature: float = 1.0):
        self.order = max(1, order)
        self.temperature = max(1e-6, float(temperature))
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.starts: Counter = Counter()

    def feed_tokens(self, tokens: List[str]):
        if len(tokens) < self.order + 1:
            return
        start = tuple(tokens[:self.order])
        self.starts[start] += 1
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            nxt = tokens[i + self.order]
            self.transitions[state][nxt] += 1

    def random_start(self) -> Tuple[str, ...]:
        if self.starts:
            return self._sample_counter(self.starts)
        if self.transitions:
            return random.choice(list(self.transitions.keys()))
        return tuple()

    def next_token(self, state: Tuple[str, ...]) -> Optional[str]:
        counter = self.transitions.get(state)
        if not counter:
            return None
        return self._sample_counter(counter)

    def _sample_counter(self, counter: Counter) -> str:
        items = list(counter.items())
        freqs = [max(1e-9, float(v)) for (_, v) in items]
        if self.temperature != 1.0:
            freqs = [pow(f, 1.0 / self.temperature) for f in freqs]
        total = sum(freqs)
        r = random.random() * total
        acc = 0.0
        for (tok, _), w in zip(items, freqs):
            acc += w
            if r <= acc:
                return tok
        return items[-1][0]

# ============================
# Generation
# ============================

END_PUNCT = [".", "!", "?", "!!", "!!!"]

def generate_sentence(model: MarkovModel, min_len: int, max_len: int,
                      stopwords: set, blacklist: set) -> str:
    if not model.transitions:
        return "(no data)"
    target_len = random.randint(min_len, max_len)
    state = model.random_start()
    sent = list(state)

    avoid = stopwords | blacklist
    if sent and any(t.lower() in avoid for t in sent):
        sent = [t for t in sent if t.lower() not in avoid]
        if len(sent) < model.order:
            state = model.random_start()
            sent = list(state)

    while len(sent) < target_len:
        state = tuple(sent[-model.order:])
        nxt = model.next_token(state)
        if not nxt:
            break
        if nxt.lower() in avoid and len(sent) < 3:
            continue
        sent.append(nxt)

    text = detokenize(sent).strip()
    if text:
        text = text[0].upper() + text[1:]
    if not any(text.endswith(p) for p in END_PUNCT):
        text += random.choice(END_PUNCT)
    return text

# ============================
# UI helpers
# ============================

HELP_LINES = [
    "[SPACE] start/stop  |  [G] generate once  |  [+/-] temperature  |  [</>] order",
    "[S] save log        |  [I] info            |  [Q] quit",
]

def _safe_addstr(stdscr, y: int, x: int, s: str, attr: int = 0):
    try:
        h, w = stdscr.getmaxyx()
        if h <= 0 or w <= 0:
            return
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        maxlen = max(0, w - x - 1)
        if maxlen <= 0:
            return
        stdscr.addstr(y, x, s[:maxlen], attr)
    except Exception:
        pass

INFO_TEXT = [
    "────────────────────────────────────────────",
    "truth_machine — v4.4  (Open Installation)",
    "",
    "This installation runs autonomously on a Raspberry Pi Zero 2W.",
    "It’s a plug-and-play generative text machine: sometimes online,",
    "sometimes not.",
    "",
    "It uses a corpus of social-media texts posted by Donald J. Trump",
    "between 2015–2017 and 2024–2025.",
    "All code is open source and can be freely used or remixed.",
    "",
    "→ GitHub / Download:",
    "   https://github.com/piotxt/truth_machine_v4_4.git",
    "",
    "→ Controls:",
    "   SPACE = start/stop | G = generate | Q = quit | I = info",
    "",
    "→ Parameters:",
    "order — how many words the machine remembers; lower = freer, higher = more coherent."
    "",
    "temp  — how unpredictable the language becomes; lower = stable, higher = wild.",
    "",

    "────────────────────────────────────────────",
    "Press any key to return."
]

def show_info(stdscr):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    start_y = max(0, (h - len(INFO_TEXT)) // 2)
    for i, line in enumerate(INFO_TEXT):
        y = start_y + i
        if y >= h:
            break
        x = max(0, (w - len(line)) // 2)
        _safe_addstr(stdscr, y, x, line)
    stdscr.refresh()
    stdscr.nodelay(False)
    stdscr.getch()
    stdscr.nodelay(True)

# ============================
# Main TUI
# ============================

def run_tui(stdscr, args, model, stopwords, blacklist, logpath):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    stdscr.keypad(True)
    STREAM_DELAY_MIN = 4.0
    STREAM_DELAY_MAX = 6.0
    streaming = False
    show_help = True
    generated = 0
    last_gen_ts = None
    next_delay = random.uniform(STREAM_DELAY_MIN, STREAM_DELAY_MAX)
    os.makedirs(os.path.dirname(logpath) or ".", exist_ok=True)
    session_header = f"\n=== Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (order={model.order}, temp={model.temperature}) ===\n"
    with open(logpath, "a", encoding="utf-8") as log:
        log.write(session_header); log.flush()
        buffer: List[str] = []
        def append_line(line: str):
            nonlocal buffer
            buffer.append(line)
            h, _w = stdscr.getmaxyx()
            max_buf = max(50, h * 3)
            if len(buffer) > max_buf:
                buffer = buffer[-max_buf:]
        def draw():
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            if h < 5 or w < 24:
                _safe_addstr(stdscr, 0, 0, "Resize the terminal, pls…")
                stdscr.refresh()
                return
            header = (f"truth_machine — order={model.order}  "
                      f"temp={model.temperature:.2f}  "
                      f"streaming={'ON' if streaming else 'OFF'}  "
                      f"generated={generated}")
            _safe_addstr(stdscr, 0, 0, header, curses.A_REVERSE)
            if show_help:
                for i, line in enumerate(HELP_LINES, start=1):
                    if i >= h - 2: break
                    _safe_addstr(stdscr, i, 0, line, curses.A_DIM)
                start_row = len(HELP_LINES) + 1
            else:
                start_row = 1
            max_lines = max(0, h - start_row - 1)
            visible = buffer[-max_lines:] if max_lines > 0 else []
            for i, line in enumerate(visible):
                y = start_row + i
                if y >= h - 1: break
                _safe_addstr(stdscr, y, 0, line)
            footer = "Press [H] help | [I] info | [Q] quit"
            _safe_addstr(stdscr, h - 1, 0, footer, curses.A_REVERSE)
            stdscr.refresh()
        draw()
        while True:
            ch = stdscr.getch()
            if ch != -1:
                if ch in (curses.KEY_RESIZE,):
                    draw(); continue
                c = chr(ch).lower() if 0 <= ch < 256 else ""
                if ch == ord(' '): streaming = not streaming
                elif c == 'q': break
                elif c == 'g':
                    phrase = generate_sentence(model, args.min_words, args.max_words, stopwords, blacklist)
                    append_line("🗣  " + phrase)
                    log.write(phrase + "\n"); log.flush()
                    generated += 1
                    last_gen_ts = time.time()
                elif c == 's': log.flush()
                elif c == 'h': show_help = not show_help
                elif c == 'i': show_info(stdscr)
                elif c == '+': model.temperature = min(5.0, round(model.temperature + 0.1, 2))
                elif c == '-': model.temperature = max(0.1, round(model.temperature - 0.1, 2))
                elif c == '<': model.order = max(1, model.order - 1)
                elif c == '>': model.order = min(8, model.order + 1)
            if streaming:
                now = time.time()
                if not last_gen_ts or now - last_gen_ts > next_delay:
                    phrase = generate_sentence(model, args.min_words, args.max_words, stopwords, blacklist)
                    append_line("🗣  " + phrase)
                    log.write(phrase + "\n"); log.flush()
                    generated += 1
                    last_gen_ts = now
                    next_delay = random.uniform(STREAM_DELAY_MIN, STREAM_DELAY_MAX)
            draw()

# ============================
# Bootstrap
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="truth_vocabulary_from_text_sm3.csv")
    ap.add_argument("--mode", choices=["vocab", "texts"], default="vocab")
    ap.add_argument("--csv-col", default="word")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--min", dest="min_words", type=int, default=8)
    ap.add_argument("--max", dest="max_words", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--stopwords", default="")
    ap.add_argument("--blacklist", default="")
    ap.add_argument("--logdir", default="output")
    ap.add_argument("--outfile", default="generated_phrases.txt")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-tui", action="store_true")
    ap.add_argument("--vocab-passes", type=int, default=32)
    ap.add_argument("--vocab-chunk-min", type=int, default=60)
    ap.add_argument("--vocab-chunk-max", type=int, default=180)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.mode == "vocab":
        words = load_from_vocab_csv(args.csv, col=args.csv_col)
        token_docs = build_synthetic_docs(words, args.order, args.vocab_passes,
                                          args.vocab_chunk_min, args.vocab_chunk_max)
    else:
        texts = load_texts_csv(args.csv, text_col=args.csv_col)
        token_docs = [tokenize(t) for t in texts]
    stopwords = load_wordlist(args.stopwords)
    blacklist = load_wordlist(args.blacklist)
    model = MarkovModel(order=args.order, temperature=args.temperature)
    for toks in token_docs:
        model.feed_tokens(toks)
    os.makedirs(args.logdir, exist_ok=True)
    outpath = os.path.join(args.logdir, args.outfile)
    if args.no_tui:
        for _ in range(10):
            print("🗣", generate_sentence(model, args.min_words, args.max_words, stopwords, blacklist))
        return
    curses.wrapper(run_tui, args, model, stopwords, blacklist, outpath)

if __name__ == "__main__":
    if "TERM" not in os.environ or not os.environ["TERM"]:
        os.environ["TERM"] = "xterm-256color"
    main()
