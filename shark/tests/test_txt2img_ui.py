import unittest
from unittest.mock import mock_open, patch

from apps.stable_diffusion.web.ui.txt2img_ui import (
    export_settings,
    load_settings,
    all_gradio_labels,
)


class TestExportSettings(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_export_settings(self, mock_json_dump, mock_file):
        test_values = ["value1", "value2", "value3"]
        expected_output = {
            "txt2img": {
                label: value
                for label, value in zip(all_gradio_labels, test_values)
            }
        }

        export_settings(*test_values)
        mock_file.assert_called_once_with("./ui/settings.json", "w")
        mock_json_dump.assert_called_once_with(
            expected_output, mock_file(), indent=4
        )

    @patch("apps.stable_diffusion.web.ui.txt2img_ui.json.load")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"txt2img": {"some_setting": "some_value"}}',
    )
    def test_load_settings_file_exists(self, mock_file, mock_json_load):
        mock_json_load.return_value = {
            "txt2img": {
                "txt2img_custom_model": "custom_model_value",
                "custom_vae": "custom_vae_value",
            }
        }

        settings = load_settings()
        self.assertEqual(settings[0], "custom_model_value")
        self.assertEqual(settings[1], "custom_vae_value")

    @patch("apps.stable_diffusion.web.ui.txt2img_ui.json.load")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_settings_file_not_found(self, mock_file, mock_json_load):
        settings = load_settings()

        default_lora_weights = "None"
        self.assertEqual(settings[4], default_lora_weights)

    @patch("apps.stable_diffusion.web.ui.txt2img_ui.json.load")
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_load_settings_key_error(self, mock_file, mock_json_load):
        mock_json_load.return_value = {}

        settings = load_settings()
        default_lora_weights = "None"
        self.assertEqual(settings[4], default_lora_weights)
