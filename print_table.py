import sys

import pandas as pd

pd.set_option('display.precision', 2)

if len(sys.argv) < 2:
  print('usage: <csv>')

tbl = pd.read_csv(sys.argv[1])
print(tbl.to_string(index=False))
