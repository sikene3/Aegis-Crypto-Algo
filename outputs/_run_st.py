import subprocess, sys
p = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run",
     r"D:\New folder (2)\Bitcoin\dashboard.py",
     "--server.headless", "true",
     "--server.port", "8765",
     "--server.address", "127.0.0.1",
     "--browser.gatherUsageStats", "false"],
    stdout=open(r"D:\New folder (2)\Bitcoin\outputs\_streamlit.out","w"),
    stderr=open(r"D:\New folder (2)\Bitcoin\outputs\_streamlit.err","w"),
)
p.wait()
