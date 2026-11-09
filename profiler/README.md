## Requirements
- scaluq
- pytest-benchmark
- matplotlib

## Profile

After installing the requirements above, enter profiler directory including cli.py

```
python3 cli.py --precision f64 --omp-threads 96 --min-nqubits 4 --max-nqubits 21
```

### Option
- `--precision`: default is f64
- `--omp-threads`: Recommend specifying number of threads (if not specified, program autodetect)
- `--min-nqubits`: default is 4
- `--max-nqubits`: default is 21

Profiler create outcome in `result` directory.

