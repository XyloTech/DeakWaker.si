import atexit
import os
import shutil
import tempfile

# Tests must never open visible windows or depend on a display: force
# headless mode (and therefore slow_mo=0) before config is imported.
os.environ["HEADLESS"] = "1"
os.environ["SLOW_MO_MS"] = "0"
# Keep each test run's persistent profile in a throwaway directory so
# tests neither pollute ./user_data nor share state across runs.
_test_profile = tempfile.mkdtemp(prefix="desk-waker-test-profile-")
os.environ["USER_DATA_DIR"] = _test_profile
atexit.register(shutil.rmtree, _test_profile, ignore_errors=True)
