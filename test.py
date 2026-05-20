from morph_kgc import VIRTStore
from rdflib import Graph
import time


"""
# Default — B11 enabled
store = VIRTStore("config.ini")

# Disabled — useful for benchmarking
store = VIRTStore("config.ini", bloom_filter=False)

# Semijoin reduction
semijoin_reduction: bool = False,

# Order BGP costs
# ── Weighted cost (tunable constants) ────────────────────────────
    α = 1.0
    β = 1.0
    γ = 0.5
    δ = 0.2

# Tune thresholds at module level if needed
import morph_kgc.sparql.virt_store as vs
vs._B11_THRESHOLD  = 128   # activate only for larger intermediate sets
vs._B11_BIT_SIZE   = 1 << 18  # 256 KB — lower false-positive rate
vs._B11_HASH_COUNT = 5
"""




store = VIRTStore('testing/c.ini')

graph = Graph(store)

start = time.perf_counter()

res = graph.query('''
PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?x ?y
WHERE {
  ?x a ub:Chair .
  ?x ub:worksFor ?y .
  ?y a ub:Department .
  ?y ub:subOrganizationOf <http://www.university0.edu> .
}
''')


for row in res:
	#print(row)
	pass
print(len(res))


elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed*1000:.3f} ms")
