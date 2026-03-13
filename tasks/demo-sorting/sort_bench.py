"""
Sorting benchmark: sort random arrays and report throughput.
This is a demo task for autoexperiments — the agent optimizes this file.
"""

import random
import time

N = 100_000
ROUNDS = 50

total_elements = 0
t0 = time.perf_counter()

for _ in range(ROUNDS):
    data = [random.randint(0, 1_000_000) for _ in range(N)]
    data.sort()
    total_elements += len(data)

elapsed = time.perf_counter() - t0
throughput = total_elements / elapsed

print("---")
print(f"throughput: {throughput:.0f}")
print(f"wall_seconds: {elapsed:.1f}")
print(f"rounds: {ROUNDS}")
print(f"array_size: {N}")
