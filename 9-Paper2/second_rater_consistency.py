"""second_rater_consistency.py: POST-HOC checks on the two returned rater sheets (2026-11-03).

Not part of the analysis fixed in docs/paper2/second_rater/PROTOCOL.md. Descriptive only.
1. Does a rater's F vs P separate leaky cells from negative controls (one-sided Fisher)?
2. F / flagged rate by the ROC-AUC printed on the sheet.
3. Self-consistency: the same mechanism appears on up to five datasets with nearly identical
   wording; share of pairs with the same rating and mean Jaccard overlap of the questions named,
   against a permutation null (the rater's own answers shuffled across items, 20000 times, seed 1).
4. Per-mechanism listing.

Usage: python 9-Paper2/second_rater_consistency.py docs/paper2/second_rater
"""
import csv, re, random, itertools, sys
from math import comb
D = sys.argv[1]
def load(f): return {r['item']: r for r in csv.DictReader(open(f"{D}/{f}", encoding='utf-8'))}
key = load('key.csv'); M = load('ratings_A.csv'); K = load('ratings_B.csv')
auc = {}
for line in open(f"{D}/sheet.md", encoding='utf-8'):
    m = re.match(r'\| ([A-E]\d) \|(.*?)\|', line)
    if m:
        a = re.search(r'ROC-AUC: (\d\.\d+)', m.group(2)); auc[m.group(1)] = float(a.group(1)) if a else None
def group(i):
    mech = key[i]['mechanism']
    if mech.startswith('negative'): return 'NEG ' + mech.split(':')[1].split(' ')[0]
    return mech.split(' ')[0] + (' absent' if 'absent' in mech else '')
items = list(key)
groups = {}
for i in items: groups.setdefault(group(i), []).append(i)

def fisher(a,b,c,d):  # one-sided, P(X>=a)
    n=a+b+c+d; r1=a+b; c1=a+c
    p=lambda x: comb(c1,x)*comb(n-c1,r1-x)/comb(n,r1)
    return sum(p(x) for x in range(a, min(r1,c1)+1))

print("== 1. Does rater A's F vs P separate leaks from clean items?")
leak=[i for i in items if not group(i).startswith('NEG')]; neg=[i for i in items if group(i).startswith('NEG')]
fl=sum(M[i]['rating']=='F' for i in leak); fn=sum(M[i]['rating']=='F' for i in neg)
print(f"leak items rated F: {fl}/{len(leak)}   clean items rated F: {fn}/{len(neg)}   one-sided Fisher p = {fisher(fl,len(leak)-fl,fn,len(neg)-fn):.4f}")
print("If rater A's P is read as 'not flagged':")
pres=[i for i in leak if 'absent' not in group(i)]
print(f"  present leaks flagged {sum(M[i]['rating']=='F' for i in pres)}/26, false alarms {fn}/14")

print("\n== 2. F rate vs reported AUC (items with a score)")
for lo,hi in [(0,0.6),(0.6,0.8),(0.8,1.01)]:
    s=[i for i in items if auc[i] is not None and lo<=auc[i]<hi]
    print(f"  AUC {lo:.1f}-{hi:.1f}: n={len(s):2d}  A F={sum(M[i]['rating']=='F' for i in s):2d}  B F/P={sum(K[i]['rating']!='N' for i in s):2d}")

def qs(r): return frozenset(q for q in r['questions'].split(';') if q)
def jac(a,b): return len(a&b)/len(a|b) if (a|b) else 1.0
def consistency(R, mapping):
    # mean over groups of pairwise (same rating) and (Jaccard of questions) between the 5 near-identical items
    rs=[];js=[]
    for g,its in mapping.items():
        if len(its)<2: continue
        for a,b in itertools.combinations(its,2):
            rs.append(R[a]['rating']==R[b]['rating']); js.append(jac(qs(R[a]),qs(R[b])))
    return sum(rs)/len(rs), sum(js)/len(js)

print("\n== 3. Self-consistency on near-identical items (same mechanism, other dataset)")
print("   (share of pairs with the same rating; mean overlap of the questions named)")
random.seed(1)
for name,R in [('first author',{i:{'rating':key[i]['first_author_rating'],'questions':key[i]['questions']} for i in items}),('rater A',M),('rater B',K)]:
    r,j=consistency(R,groups)
    # permutation null: shuffle this rater's own answers across items
    vals=[R[i] for i in items]; hits_r=hits_j=0; N=20000
    for _ in range(N):
        random.shuffle(vals); P=dict(zip(items,vals)); rr,jj=consistency(P,groups)
        hits_r+= rr>=r; hits_j+= jj>=j
    print(f"  {name:12s} same rating {r:.2f} (p={hits_r/N:.4f})   question overlap {j:.2f} (p={hits_j/N:.4f})")

print("\n== 4. Per group, what each rater wrote")
for g,its in sorted(groups.items()):
    print(f"  {g}")
    for i in its:
        print(f"     {i} AUC={auc[i]}  author={key[i]['first_author_rating']}:{key[i]['questions']:7s} A={M[i]['rating']}:{M[i]['questions']:9s} B={K[i]['rating']}:{K[i]['questions']}")
