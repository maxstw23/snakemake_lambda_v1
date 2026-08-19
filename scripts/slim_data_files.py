"""Make GitHub-sized copies of the raw ROOT inputs under ``data/``.

The upstream productions carry far more than this pipeline ever opens: the
Lambda/Xi mass and v1 histograms are booked in 30 pT bins covering 0-3 GeV/c
while ``config.yaml`` selects only 0.4-1.8 GeV/c, the piKp v2 files carry ~230
QA objects of which ``Finish_v1_tof_eff.C`` reads 26, and the efficiency files
carry 147 objects of which ``calculate_lambda_eff.cpp`` reads 18.  Dropping
what is never read takes ``data/`` from 3.7 GB to ~1.2 GB, with the largest file
going from 164 MB to ~49 MB, so every file clears GitHub's 100 MB limit and the
tree can be tracked without LFS.  Nothing that is kept is altered in any way.

    python3 scripts/slim_data_files.py --out backup/data_slim
    python3 scripts/slim_data_files.py --out backup/data_slim --check   # verify

The slim tree is a drop-in replacement for ``data/`` for every rule in the
Snakefile at the current ``config.yaml`` settings.  Three things need the raw
originals: widening ``pt_lo``/``pt_hi`` past 0.4-1.8 GeV/c, adding ``a1`` to
``flows``, and ``scripts/check_lambda_reco.py`` (which scans all 30 pT bins).
"""
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import shutil
import sys
import time
from glob import glob

import numpy as np
import uproot

# --- data/result*.root and data/sys_tag_*/result*.root -----------------------
STRANGE = r"(Lambda|Lambdabar|Xi|Xibar)"
# The (cen, ybin, ptbin) arrays: 4 species x 9 cen x 20 y x 30 pT, ~95% of the file.
ARRAY_RE = re.compile(rf"^h{STRANGE}(?:M_cen_y_pt|_EPD_v\d+_pt)_(\d+)_(\d+)_(\d+)$")
# Dropped outright:
#   hgp*/hgKp*      PID QA (dE/dx, 1/beta, m^2 vs p), 2.5-3.6 MB per file;
#   *_EPD_a1_pt_*   the a1 twin of the v1 arrays. config.yaml sets flows: [v1], so
#                   combine_lambda_*.cpp never asks for a1, and only 5 of the 28
#                   productions have them at all (14.6 GeV and the 19.6 GeV set) --
#                   not enough energies to run an a1 analysis from anyway. They are
#                   69-74 MB in exactly the files that are already the largest, which
#                   is the difference between the biggest slim file being 49 MB and
#                   it being 118 MB, i.e. over GitHub's 100 MB limit.
MAIN_DROP_RE = re.compile(rf"^hg(p|Kp)|^h{STRANGE}_EPD_a\d+_pt_\d+_\d+_\d+$")

# --- data/eff/result*_{lambda,lambdabar}_exp_*.root --------------------------
# calculate_lambda_eff.cpp reads only the 2D (pT, y) MC and reco spectra.
EFF_KEEP_RE = re.compile(r"^h(MCParPtY|KFPRecoParPtY)_\d+$")

# --- data/v1_piKp/{energy}/{particle}/cen*.v2_pion.root ----------------------
# Everything Finish_v1_tof_eff.C calls f->Get() on.
PIKP_KEEP = {
    "Hist_Pt", "Hist_Pt_TOF",                            # TOF matching efficiency
    "EtaPtDist", "EtaPtDistp", "EtaPtDistn",             # yields
    "Hist_Yield_cos1", "Hist_Yield_sin1",                # event-plane weights
    "Hist_Yield_cos2", "Hist_Yield_sin2",
    "Hist_cos_EPD",                                      # EPD resolution
} | {
    f"p_{flow}_{ew}_{cs}_obs{i}"
    for flow in ("v1", "a1") for ew in ("e", "w") for cs in ("c", "s") for i in (1, 2)
}

COMPRESSION = {
    "zlib": lambda: uproot.ZLIB(9),   # what ROOT writes, at max level
    "lzma": lambda: uproot.LZMA(5),   # marginally smaller here, ~2x slower
    "lz4": lambda: uproot.LZ4(4),     # fastest to read back
}


def main_keeper(ptbin_lo: int, ptbin_hi: int):
    """Keep every object except array bins outside the analysis pT window."""
    def keep(name: str) -> bool:
        if MAIN_DROP_RE.match(name):
            return False
        m = ARRAY_RE.match(name)
        if m is None:
            return True
        return ptbin_lo <= int(m.group(4)) <= ptbin_hi
    return keep


def classify(relpath: str):
    """Return the keep-predicate for a file, or None to copy it verbatim."""
    parts = relpath.replace(os.sep, "/").split("/")
    if parts[0] == "eff":
        return lambda name: bool(EFF_KEEP_RE.match(name))
    if parts[0] == "v1_piKp":
        return lambda name: name in PIKP_KEEP
    return None  # caller substitutes the main keeper


def slim(src: str, dst: str, keep, compression) -> int:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    n = 0
    with uproot.open(src) as fin, uproot.recreate(dst, compression=compression) as fout:
        for key in fin.keys(cycle=False, recursive=False):
            if keep(key):
                fout[key] = fin[key]
                n += 1
    # Carry the original's timestamps across. Snakemake decides what is stale by
    # mtime, so a slim tree written "now" would look newer than every result in
    # result/ and re-trigger the whole pipeline -- ~200 of those jobs are fits that
    # take minutes to hours each. (shutil.copy2 does this for the verbatim files.)
    st = os.stat(src)
    os.utime(dst, (st.st_atime, st.st_mtime))
    return n


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_verbatim(src: str, dst: str):
    """For the files that are copied rather than slimmed, demand byte equality."""
    same = _sha256(src) == _sha256(dst)
    return 1, [] if same else [("<file>", "sha256")]


def check(src: str, dst: str):
    """Every object in dst must hold the same contents as the one in src."""
    bad = []
    with uproot.open(src) as a, uproot.open(dst) as b:
        keys = b.keys(cycle=False, recursive=False)
        for k in keys:
            x, y = a[k], b[k]
            if x.classname != y.classname:
                bad.append((k, "classname"))
                continue
            compared = False
            for m in ("values", "errors"):
                # TGraph/TGraphErrors/TF1 do not expose these; they are covered by
                # the member sweep below instead.
                try:
                    va = np.nan_to_num(getattr(x, m)())
                    vb = np.nan_to_num(getattr(y, m)())
                except (AttributeError, TypeError):
                    continue
                compared = True
                if not np.array_equal(va, vb):
                    bad.append((k, m))
            for member in ("fBinEntries", "fSumw2", "fEntries",
                           "fX", "fY", "fEX", "fEY", "fNpoints"):
                try:
                    va, vb = x.member(member), y.member(member)
                except Exception:
                    continue
                compared = True
                if not np.array_equal(np.asarray(va), np.asarray(vb)):
                    bad.append((k, member))
            if not compared:
                bad.append((k, "no comparable payload"))
    return len(keys), bad


def _is_verbatim(rel: str) -> bool:
    """Model files carry TGraphErrors/TF1 no keep-list applies to; copy them whole."""
    return rel.replace(os.sep, "/").startswith("model/")


class _Task:
    """One file's worth of work, as a picklable callable for ProcessPoolExecutor."""

    def __init__(self, src_root, out_root, do_check, compression, ptbin_lo, ptbin_hi):
        self.src_root = src_root
        self.out_root = out_root
        self.do_check = do_check
        self.compression = compression
        self.ptbin_lo = ptbin_lo
        self.ptbin_hi = ptbin_hi

    def __call__(self, src):
        rel = os.path.relpath(src, self.src_root)
        dst = os.path.join(self.out_root, rel)

        if self.do_check:
            if not os.path.exists(dst):
                return f"MISSING  {dst}", 0, 0, 1
            n, bad = (check_verbatim if _is_verbatim(rel) else check)(src, dst)
            line = (f"{'OK ' if not bad else 'FAIL'}  {n:6d} objs  {dst}"
                    + (f"   {bad[:3]}" if bad else ""))
            return line, 0, 0, int(bool(bad))

        t0 = time.time()
        if _is_verbatim(rel):
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)
            n = -1
        else:
            keep = classify(rel) or main_keeper(self.ptbin_lo, self.ptbin_hi)
            n = slim(src, dst, keep, COMPRESSION[self.compression]())
        s_in, s_out = os.path.getsize(src), os.path.getsize(dst)
        tag = "copied" if n < 0 else f"{n:6d} objs"
        flag = "  <-- nothing matched, stale production?" if n == 0 else ""
        line = (f"{s_in/1e6:7.1f} -> {s_out/1e6:6.1f} MB  {tag}  "
                f"{time.time()-t0:5.0f}s  {dst}{flag}")
        return line, s_in, s_out, 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data", help="source tree (default: data)")
    ap.add_argument("--out", required=True, help="destination tree for the slim copies")
    ap.add_argument("--compression", default="zlib", choices=sorted(COMPRESSION))
    ap.add_argument("--pt_lo", type=float, default=0.4, help="config.yaml pt_lo")
    ap.add_argument("--pt_hi", type=float, default=1.8, help="config.yaml pt_hi")
    ap.add_argument("--check", action="store_true",
                    help="verify existing slim copies against the source instead of writing")
    ap.add_argument("--only", default=None,
                    help="restrict to source paths containing this substring")
    ap.add_argument("--jobs", type=int, default=1,
                    help="slim this many files in parallel (one process each)")
    args = ap.parse_args()

    # Same bin indices combine_lambda_{with,without}_eff.cpp derives from the cuts.
    ptbin_lo, ptbin_hi = int(args.pt_lo * 10), int(args.pt_hi * 10) - 1

    files = sorted(glob(os.path.join(args.src, "**", "*.root"), recursive=True))
    if args.only:
        files = [f for f in files if args.only in f]
    if not files:
        sys.exit(f"no *.root under {args.src}/")

    tot_src = tot_dst = 0
    failures = 0
    task = _Task(args.src, args.out, args.check, args.compression, ptbin_lo, ptbin_hi)

    def report(results):
        """Print each result as it lands and accumulate the totals."""
        nonlocal tot_src, tot_dst, failures
        for line, s_in, s_out, fail in results:
            print(line, flush=True)
            tot_src += s_in
            tot_dst += s_out
            failures += fail

    if args.jobs > 1:
        # Slowest files first, so the tail of the run is not one 5-minute straggler.
        files = sorted(files, key=os.path.getsize, reverse=True)
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            report(pool.map(task, files))
    else:
        report(task(src) for src in files)

    if args.check:
        print(f"\n{len(files)} files checked, {failures} problem(s)")
        sys.exit(1 if failures else 0)
    print(f"\n{len(files)} files: {tot_src/1e9:.2f} GB -> {tot_dst/1e9:.2f} GB "
          f"({tot_dst/tot_src*100:.0f}%)")


if __name__ == "__main__":
    main()
