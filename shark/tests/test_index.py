import os
import pytest
import tempfile
from unittest import mock
from apps.stable_diffusion.web.index import cleanup_mei_folders


# Test for removing temporary _MEI folders on windows
def test_cleanup_mei_folders_windows():
    # Setting up the test environment for Windows
    with mock.patch('sys.platform', 'win32'):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_temp_dir = os.path.join(temp_dir, 'Temp')
            os.makedirs(temp_temp_dir)

            # Creating a fictitious _MEI directory
            with mock.patch.dict('os.environ', {'LOCALAPPDATA': temp_dir}):
                mei_folder = os.path.join(temp_temp_dir, '_MEI12345')
                os.makedirs(mei_folder)

                cleanup_mei_folders()
                assert not os.path.exists(mei_folder)


# Test for removing temporary folders at unsupported OS
def test_cleanup_mei_folders_unsupported_os():
    with mock.patch('sys.platform', 'unsupported_os'):
        with pytest.warns(UserWarning) as record:
            cleanup_mei_folders()

        assert "Temporary files weren't deleted due to an unsupported OS" in str(record.list[0].message)