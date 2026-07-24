```bash
conda activate retorm
python runner.py --schemas 3 --queries 20 --seed 42 --no-z3 --verbose
python runner.py --schemas 10 --queries 50 --tables 3 --cols 4
python runner.py --schemas 1200 --queries 50 --tables 3 --cols 5
python runner.py --schemas 600 --queries 80 --tables 4 --cols 6
```
