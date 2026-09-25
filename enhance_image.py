"""PureScale 4.0 CLI Entrypoint Wrapper.

Provides complete backward compatibility while delegating directly to the
modern purescale.cli module with autonomous diagnostics and advanced computer vision.
"""

import sys
from purescale.cli import main

if __name__ == "__main__":
    sys.exit(main())
