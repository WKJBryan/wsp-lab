import subprocess, sys, os
# reuse testsuite but point at the candidate dir
os.environ['CAND'] = sys.argv[1]
