import glob, json, sys
base = '/home/me/Documents/strata/data/rp-opt/accept-attn'
arms = sys.argv[1:] or ['a1', 'k1', 'a2']
names = {1: 'docs 1911', 2: 'factual 70', 3: 'code 1532'}
print('%-6s %-12s %6s %8s %8s %7s' % ('arm', 'request', 'gen', 'drafts', 'acc', 'acc %'))
tot = {}
for a in arms:
    ds = as_ = 0
    for n in (1, 2, 3):
        d = json.load(open('%s/mtp4-%s-%d.json' % (base, a, n)))
        t = d.get('timings', {})
        dn, da = t.get('draft_n', 0), t.get('draft_n_accepted', 0)
        ds += dn; as_ += da
        print('%-6s %-12s %6d %8d %8d %6.1f%%' % (a, names[n], t.get('predicted_n', 0), dn, da, 100.0 * da / dn if dn else 0))
    tot[a] = (ds, as_)
    print('%-6s %-12s %6s %8d %8d %6.1f%%' % (a, 'all three', '', ds, as_, 100.0 * as_ / ds if ds else 0))
print()
for a, (ds, as_) in tot.items():
    print('aggregate %-3s %d/%d = %.2f %%' % (a, as_, ds, 100.0 * as_ / ds if ds else 0))
