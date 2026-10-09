"""CLI: python run.py BTC [ETH SOL ...]  -> reports/<SYM>.md and reports/<SYM>.json"""
import os
import sys

from bwpredict.chart import fan_chart
from bwpredict.report import run_asset, to_json

if __name__ == "__main__":
    syms = sys.argv[1:] or ["BTC"]
    os.makedirs("reports", exist_ok=True)
    for sym in syms:
        md, js = run_asset(sym.upper())
        with open(f"reports/{sym.upper()}.md", "w") as fh:
            fh.write(md)
        with open(f"reports/{sym.upper()}.json", "w") as fh:
            fh.write(to_json(js))
        fan_chart(sym.upper(), f"reports/{sym.upper()}.json", f"reports/{sym.upper()}_forecast.png")
        print(md.split("## Horizon")[0])
