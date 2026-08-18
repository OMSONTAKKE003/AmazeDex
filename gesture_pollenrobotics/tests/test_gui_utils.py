"""
Unit tests for pure utility functions in amazing_hand_gui.py.

No hardware connection or GUI display is needed; only isolated helper
functions are exercised.
"""
import pytest
import hand_logic
import amazing_hand_gui as gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    """Redirect CONFIG_FILE and DATA_DIR to a temporary directory.

    Returns the (config_file_path, data_dir_path) tuple so individual
    tests can pre-populate or inspect the YAML file.
    """
    cf = tmp_path / "hand_config.yaml"
    monkeypatch.setattr(hand_logic, "CONFIG_FILE", cf)
    monkeypatch.setattr(hand_logic, "DATA_DIR", tmp_path)
    return cf, tmp_path


# ---------------------------------------------------------------------------
# validate_name
# ---------------------------------------------------------------------------

class TestValidateName:
    def test_simple_word(self):
        valid, _ = gui.validate_name("open")
        assert valid

    def test_underscore_and_digits(self):
        valid, _ = gui.validate_name("my_pose_01")
        assert valid

    def test_spaces_in_middle_allowed(self):
        # Spaces are not in the forbidden list; only leading/trailing are stripped
        valid, _ = gui.validate_name("my pose")
        assert valid

    def test_empty_string_rejected(self):
        valid, msg = gui.validate_name("")
        assert not valid
        assert msg

    def test_whitespace_only_rejected(self):
        valid, msg = gui.validate_name("   ")
        assert not valid
        assert msg

    def test_exactly_50_chars_accepted(self):
        valid, _ = gui.validate_name("a" * 50)
        assert valid

    def test_51_chars_rejected(self):
        valid, msg = gui.validate_name("a" * 51)
        assert not valid
        assert "50" in msg or "long" in msg.lower()

    @pytest.mark.parametrize("forbidden", [
        ":", "{", "}", "[", "]", ",", "&", "*", "#", "?",
        "|", "-", "<", ">", "=", "!", "%", "@", "`", '"', "'",
    ])
    def test_forbidden_character_rejected(self, forbidden):
        valid, msg = gui.validate_name(f"pose{forbidden}x")
        assert not valid
        assert "forbidden" in msg.lower()

    def test_null_byte_rejected(self):
        valid, _ = gui.validate_name("pose\x00name")
        assert not valid

    def test_tab_character_rejected(self):
        # ASCII 9 (tab) < 32 → control character
        valid, _ = gui.validate_name("pose\tname")
        assert not valid

    def test_newline_rejected(self):
        valid, _ = gui.validate_name("pose\nname")
        assert not valid


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_value_within_range(self):
        assert gui.clamp(5, 0, 10) == 5

    def test_value_below_min_clamped(self):
        assert gui.clamp(-5, 0, 10) == 0

    def test_value_above_max_clamped(self):
        assert gui.clamp(15, 0, 10) == 10

    def test_value_at_min_unchanged(self):
        assert gui.clamp(0, 0, 10) == 0

    def test_value_at_max_unchanged(self):
        assert gui.clamp(10, 0, 10) == 10

    def test_float_values(self):
        assert gui.clamp(1.5, 1.0, 2.0) == pytest.approx(1.5)

    def test_negative_range_lower(self):
        assert gui.clamp(-50, -40, 40) == -40

    def test_negative_range_upper(self):
        assert gui.clamp(50, -40, 40) == 40


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_returns_empty_structure(self, config_path):
        config = gui.load_config()
        assert config == {"poses": {}, "sequences": {}}

    def test_empty_yaml_file_returns_structure(self, config_path):
        cf, _ = config_path
        cf.write_text("")
        config = gui.load_config()
        assert "poses" in config
        assert "sequences" in config

    def test_valid_yaml_loaded_correctly(self, config_path):
        cf, _ = config_path
        cf.write_text(
            "poses:\n"
            "  open:\n"
            "    positions: [0, 0, 0, 0, 0, 0, 0, 0]\n"
        )
        config = gui.load_config()
        assert "open" in config["poses"]
        assert config["poses"]["open"]["positions"] == [0] * 8

    def test_missing_sequences_key_added_automatically(self, config_path):
        cf, _ = config_path
        cf.write_text(
            "poses:\n"
            "  open:\n"
            "    positions: [0, 0, 0, 0, 0, 0, 0, 0]\n"
        )
        config = gui.load_config()
        assert "sequences" in config

    def test_missing_poses_key_added_automatically(self, config_path):
        cf, _ = config_path
        cf.write_text("sequences: {}\n")
        config = gui.load_config()
        assert "poses" in config

    def test_malformed_yaml_returns_empty(self, config_path):
        cf, _ = config_path
        cf.write_text("this: is: {malformed\n")
        config = gui.load_config()
        assert config == {"poses": {}, "sequences": {}}


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

class TestSaveConfig:
    def test_save_returns_true(self, config_path):
        result = gui.save_config({"poses": {}, "sequences": {}})
        assert result is True

    def test_file_is_created(self, config_path):
        cf, _ = config_path
        gui.save_config({"poses": {"open": {"positions": [0] * 8}}, "sequences": {}})
        assert cf.exists()

    def test_positions_written_as_inline_array(self, config_path):
        cf, _ = config_path
        gui.save_config({"poses": {"open": {"positions": [0] * 8}}, "sequences": {}})
        assert "positions: [0, 0, 0, 0, 0, 0, 0, 0]" in cf.read_text()

    def test_negative_positions_preserved(self, config_path):
        positions = [-40, -10, 0, 10, 20, 30, 40, 50]
        gui.save_config({"poses": {"neg": {"positions": positions}}, "sequences": {}})
        loaded = gui.load_config()
        assert loaded["poses"]["neg"]["positions"] == positions

    def test_mixed_positions_inline_format(self, config_path):
        cf, _ = config_path
        positions = [0, 110, 0, 110, 0, 110, 0, 110]
        gui.save_config({"poses": {"alt": {"positions": positions}}, "sequences": {}})
        content = cf.read_text()
        assert "positions: [0, 110, 0, 110, 0, 110, 0, 110]" in content

    def test_round_trip_poses_and_sequences(self, config_path):
        original = {
            "poses": {
                "open": {"positions": [0] * 8},
                "close": {"positions": [110] * 8},
            },
            "sequences": {
                "demo": {
                    "steps": [
                        "open:3,3,3,3,3,3,3,3|2.0s",
                        "close:3,3,3,3,3,3,3,3|2.0s",
                    ]
                }
            },
        }
        gui.save_config(original)
        loaded = gui.load_config()

        assert set(loaded["poses"].keys()) == {"open", "close"}
        assert loaded["poses"]["open"]["positions"] == [0] * 8
        assert loaded["poses"]["close"]["positions"] == [110] * 8
        assert "demo" in loaded["sequences"]
        assert len(loaded["sequences"]["demo"]["steps"]) == 2
