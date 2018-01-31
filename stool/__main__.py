import sys
if sys.argv[0].endswith("__main__.py"):
    sys.argv[0] = "python -m stool"

__stool = True

from .main import main

main()
