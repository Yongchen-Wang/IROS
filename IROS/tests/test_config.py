from pathlib import Path

from iros.navigation.config import NavigationConfig, load_config, save_config

def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "navigation.json"
    expected = NavigationConfig(
        robot_radius_px=12.5,
        targets_maze={"A": (4.0, 8.0)},
        max_arm_switches=0,
    )

    save_config(expected, path)
    actual = load_config(path)

    assert actual.robot_radius_px == 12.5
    assert actual.targets_maze == {"A": (4.0, 8.0)}
    assert actual.max_arm_switches == 0


def test_example_config_has_required_geometry() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "navigation" / "example.json"
    config = load_config(config_path)

    assert config.robot_radius_px is not None
    assert config.robot_radius_px > 0
    if config.enable_ump_rod_detection:
        assert config.ump_rod_width_maze is not None
        assert config.ump_rod_width_maze > 0
        assert config.ump_rod_length_maze is not None
        assert config.ump_rod_length_maze > 0

def test_point_start_config_disables_arm_switching() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "navigation" / "point_start.json"
    config = load_config(config_path)

    assert config.max_arm_switches == 0

def test_missing_config_returns_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.json")
    assert config.path_resolution == 2.0
