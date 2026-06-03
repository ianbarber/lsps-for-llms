import json, math

# ---- Load ----
P = json.load(open("runs/agent/synth_power.json"))
E = json.load(open("runs/agent/synth_ceager.json"))

def index(rows):
    d = {}
    for r in rows:
        d[(r["task"], r["seed"])] = r
    return d

cond = {
    "A":       index(P["rows"]["A"]),
    "C-lazy":  index(P["rows"]["C"]),
    "C-eager": index(E["rows"]["C"]),
    "D-tuned": index(P["rows"]["D"]),
}

# sanity: counts & key sets
keys = {k: set(v.keys()) for k,v in cond.items()}
allkeys = set.union(*keys.values())
common = set.intersection(*keys.values())
print("== sanity ==")
for k,v in cond.items():
    print(f"{k:8s} n={len(v)}  resolved={sum(r['resolved'] for r in v.values())}")
print("union keys:", len(allkeys), "common keys:", len(common))
print("all four share identical key sets:", all(keys[k]==common for k in cond))

# ---- Wilson 95% CI ----
def wilson(k, n, z=1.959963984540054):
    if n==0: return (0,0,0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/denom
    return p, center-half, center+half

print("\n== 1. Resolve rate (n/84) + 95% Wilson CI ==")
order = ["C-eager","C-lazy","A","D-tuned"]
res = {}
for k in order:
    n = len(cond[k]); kk = sum(r["resolved"] for r in cond[k].values())
    p, lo, hi = wilson(kk, n)
    res[k] = kk
    print(f"{k:8s}: {kk}/{n} = {p:.3f}  Wilson95% [{lo:.3f}, {hi:.3f}]")

# ---- McNemar exact ----
def mcnemar(X, Y, common):
    # b: X resolved, Y not ; c: Y resolved, X not
    b=c=0
    for key in common:
        rx = cond[X][key]["resolved"]; ry = cond[Y][key]["resolved"]
        if rx and not ry: b+=1
        elif ry and not rx: c+=1
    n = b+c
    m = min(b,c)
    # exact two-sided binomial p
    if n==0:
        p = 1.0
    else:
        tail = sum(math.comb(n,i) for i in range(0,m+1)) / (2**n)
        p = min(1.0, 2*tail)
    return b,c,p

print("\n== 2. Paired McNemar exact (two-sided) over", len(common), "shared (task,seed) units ==")
pairs = [("D-tuned","C-eager"),("D-tuned","A"),("C-eager","A"),
         ("C-eager","C-lazy"),("D-tuned","C-lazy"),("A","C-lazy")]
print(f"{'X':8s} vs {'Y':8s}  b(Xonly) c(Yonly)   p_exact")
for X,Y in pairs:
    b,c,p = mcnemar(X,Y,common)
    print(f"{X:8s} vs {Y:8s}   b={b:3d}    c={c:3d}    p={p:.4g}")

# ---- 3. Matched-pair efficiency on units resolved by BOTH ----
def wilcoxon_signed_rank(diffs):
    # two-sided exact-ish; use normal approx with continuity + tie correction
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n==0: return n, None, None
    absd = sorted((abs(d), (1 if d>0 else -1)) for d in nz)
    # rank with ties averaged
    ranks = [0.0]*n
    i=0
    vals=[a for a,_ in absd]
    while i<n:
        j=i
        while j<n and vals[j]==vals[i]:
            j+=1
        avg = (i+1+j)/2.0  # average rank (1-based)
        for k in range(i,j):
            ranks[k]=avg
        i=j
    Wpos = sum(rk for rk,(a,s) in zip(ranks,absd) if s>0)
    Wneg = sum(rk for rk,(a,s) in zip(ranks,absd) if s<0)
    W = min(Wpos, Wneg)
    mu = n*(n+1)/4.0
    # tie correction
    from collections import Counter
    cnt = Counter(vals)
    tiecorr = sum(t**3 - t for t in cnt.values())
    sigma = math.sqrt(n*(n+1)*(2*n+1)/24.0 - tiecorr/48.0)
    if sigma==0: return n, Wpos, None
    z = (W - mu + 0.5*(1 if W<mu else -1))/sigma  # continuity toward mu
    # two-sided p via normal
    p = math.erfc(abs(z)/math.sqrt(2))
    return n, (Wpos, Wneg), p

def sign_test(diffs):
    pos = sum(1 for d in diffs if d>0)
    neg = sum(1 for d in diffs if d<0)
    n = pos+neg
    m = min(pos,neg)
    if n==0: return pos,neg,1.0
    tail = sum(math.comb(n,i) for i in range(0,m+1))/(2**n)
    return pos, neg, min(1.0, 2*tail)

print("\n== 3. Matched-pair efficiency (units resolved by BOTH) ==")
eff_pairs = [("D-tuned","C-lazy"),("D-tuned","A"),("D-tuned","C-eager")]
for X,Y in eff_pairs:
    both = [key for key in common if cond[X][key]["resolved"] and cond[Y][key]["resolved"]]
    npairs = len(both)
    for metric in ["n_tests","out_tokens"]:
        xs = [cond[X][key][metric] for key in both]
        ys = [cond[Y][key][metric] for key in both]
        mx = sum(xs)/npairs if npairs else float('nan')
        my = sum(ys)/npairs if npairs else float('nan')
        diffs = [cond[X][key][metric]-cond[Y][key][metric] for key in both]  # X - Y ; negative => X cheaper
        nnz, Wstat, pw = wilcoxon_signed_rank(diffs)
        pos,neg,psign = sign_test(diffs)
        print(f"{X} vs {Y} [{metric}] n_pairs={npairs}  mean {X}={mx:.2f}  mean {Y}={my:.2f}  meanDiff(X-Y)={sum(diffs)/npairs:+.2f}")
        print(f"     Wilcoxon: n_nonzero={nnz} W={Wstat} p={pw if pw is None else round(pw,4)} ; Sign: +{pos}/-{neg} p={round(psign,4)}")
