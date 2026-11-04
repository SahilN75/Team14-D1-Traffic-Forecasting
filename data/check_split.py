import sys, csv, hashlib
import numpy as np
f, summ = sys.argv[1], sys.argv[2]
raw = open(f, "rb").read()
print("BOM:", raw[:3] == b"\xef\xbb\xbf", "| sha256", hashlib.sha256(raw).hexdigest())
rows = list(csv.reader(raw.decode("utf-8").splitlines()))
print("header:", rows[0]); rows = rows[1:]
print("rows:", len(rows), "| unique sample_id:", len({r[0] for r in rows}), "| unit==sample:", all(r[0] == r[1] for r in rows),
      "| labels empty:", all(r[3] == "" for r in rows), "| reasons empty:", all(r[4] == "" for r in rows))
N = 34272; b1, b2 = int(.7 * N), int(.8 * N)
part = {}
for r in rows: part.setdefault(r[2], []).append(int(r[0]))
for p, ids in part.items():
    ids = np.array(sorted(ids)); d = np.unique(np.diff(ids))
    print(f"{p:5s} n={len(ids)} first={ids[0]} last={ids[-1]} stride={d.tolist()}")
tr, va, te = (np.array(sorted(part[k])) for k in ("train", "val", "test"))
print("\nAs LAST INPUT ROW s (history s-11..s, targets s+3,s+6,s+12):")
print(" train last target", tr[-1] + 12, "< b1", b1, "->", tr[-1] + 12 < b1, "| first history row", tr[0] - 11)
print(" val first/last target", va[0] + 3, va[-1] + 12, "inside [b1,b2):", va[0] + 3 >= b1 and va[-1] + 12 < b2, "| first history row", va[0] - 11, "(crosses boundary: %s)" % (va[0] - 11 < b1))
print(" test first/last target", te[0] + 3, te[-1] + 12, "inside [b2,N):", te[0] + 3 >= b2 and te[-1] + 12 < N)
print(" vs Annex D4 (id = last input row): train starts 11 ->", "file starts", tr[0], "(", tr[0] - 11, "step later )", "| val/test start at boundary rows (accepted):", va[0] == b1, te[0] == b2)
print("\nAs ROW AFTER LAST INPUT s'=id (last input = id-1):")
print(" train last input", tr[-1] - 1, "first", tr[0] - 1, "-> annex start 11 matches:", tr[0] - 1 == 11)
print(" val first last-input", va[0] - 1, "(annex-eligible 23987 or accepted 23990; this is neither)")
print(" test first last-input", te[0] - 1, "(annex 27414 / accepted 27417; neither)")
print("\nsummary file:")
for r in csv.reader(open(summ, encoding="utf-8").read().splitlines()): print(" ", r[:4], "hash ok:" , r[-1] == hashlib.sha256(raw).hexdigest() if r[0] != "file" else "")
