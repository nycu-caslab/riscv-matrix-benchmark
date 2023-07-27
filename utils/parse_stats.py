import sys
import os
import re
import numpy as np

root_dir=""
if len(sys.argv) > 1:
  root_dir=sys.argv[1]






stats_dir = os.listdir(root_dir)

print(stats_dir)

numCycle_ls = np.zeros((257, 257, 257), dtype=np.int32)
valid_list = []

for stat_dir in stats_dir:

  pat = r".*-[0-9]+-M([0-9]+)-K([0-9]+)-N([0-9]+)\.stats"
  res = re.search(pat, stat_dir)

  print(res.groups())

  stat_path = os.path.join(root_dir, stat_dir, 'm5out', 'stats.txt')

  f = None

  
  stat_path = os.path.join(root_dir, stat_dir, 'm5out', 'stats.txt')
  f = open(stat_path, "r")

  lines = f.readlines()

  numCycles_line = lines[14]

  numCycles = re.search(r"\S+\s+([0-9]+)\s+.*", numCycles_line).groups()[0]

  print(numCycles)
  m, k, n = res.groups()
  m = int(m)
  k = int(k)
  n = int(n)
  # numCycle_ls.append((m, k, n, numCycles))
  numCycle_ls[m, k, n] = numCycles
  valid_list.append((m, k, n))


# numCycle_ls.sort(key=lambda x: x[0])


with open("mxt-perf.txt", "w") as file:

  for valid_mkn in valid_list:
    m, k, n = valid_mkn
    cycs = numCycle_ls[m, k, n]
    file.write(f"{m:3} {k:3} {n:3} {cycs:10}\n")
  # for m in range(16, 257, 32):
  #   for k in range(16, 257, 32):
  #     for n in range(16, 257, 32):
  #       cycs = numCycle_ls[m, k, n]
  #       file.write(f"{m:3} {k:3} {n:3} {cycs:10}\n")
  #   file.write("\n\n")

  # for m, k, n, cycs in numCycle_ls:
  #   file.write(f"{m:3} {k:3} {n:3} {cycs:10}\n")


