from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

StartRegionCenter = Tuple[float, float]

@dataclass
class CircularStartRegion:

    center: StartRegionCenter
    radius: float

@dataclass
class RectStartRegion:

    center: StartRegionCenter
    width: float
    height: float

StartRegion = Union[CircularStartRegion, RectStartRegion]

@dataclass
class NavigationConfig:

    robot_radius_px: Optional[float] = None

    ump_rod_width_maze: Optional[float] = None
    ump_rod_length_maze: Optional[float] = None

    start_region: Optional[StartRegion] = None

    safety_margin_weight: float = 5.0
    path_resolution: float = 2.0

    path_smoothing_enabled: bool = False
    path_smoothing_factor: float = 0.3                                                        

    center_preference_alpha: float = 2.0

    use_exponential_cost: bool = True

    effective_radius_px_override: Optional[float] = None

    disable_ump_mask_for_planning: bool = False

    n_start_samples: int = 5

    enable_ump_rod_detection: bool = False

    targets_maze: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    max_arm_switches: int = 1

    output_dir: Path = field(default_factory=lambda: Path("outputs") / "navigation")

    def to_dict(self) -> Dict[str, Any]:

        def serialize_start_region(region: Optional[StartRegion]) -> Optional[Dict[str, Any]]:
            if region is None:
                return None
            if isinstance(region, CircularStartRegion):
                return {
                    "type": "circular",
                    "center": list(region.center),
                    "radius": region.radius,
                }
            if isinstance(region, RectStartRegion):
                return {
                    "type": "rectangular",
                    "center": list(region.center),
                    "width": region.width,
                    "height": region.height,
                }
            raise TypeError(f"Unsupported start_region type: {type(region)}")

        return {
            "robot_radius_px": self.robot_radius_px,
            "ump_rod_width_maze": self.ump_rod_width_maze,
            "ump_rod_length_maze": self.ump_rod_length_maze,
            "start_region": serialize_start_region(self.start_region),
            "safety_margin_weight": self.safety_margin_weight,
            "path_resolution": self.path_resolution,
            "path_smoothing_enabled": self.path_smoothing_enabled,
            "path_smoothing_factor": self.path_smoothing_factor,
            "effective_radius_px_override": self.effective_radius_px_override,
            "disable_ump_mask_for_planning": self.disable_ump_mask_for_planning,
            "n_start_samples": self.n_start_samples,
            "enable_ump_rod_detection": self.enable_ump_rod_detection,
            "targets_maze": {k: list(v) for k, v in self.targets_maze.items()},
            "max_arm_switches": self.max_arm_switches,
            "output_dir": str(self.output_dir),
            "center_preference_alpha": self.center_preference_alpha,
            "use_exponential_cost": self.use_exponential_cost,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NavigationConfig":

        def parse_start_region(raw: Optional[Dict[str, Any]]) -> Optional[StartRegion]:
            if raw is None:
                return None

            r_type = raw.get("type", "circular")
            center = tuple(raw.get("center", [0.0, 0.0]))                            

            if r_type == "circular":
                return CircularStartRegion(center=center, radius=float(raw.get("radius", 20.0)))
            if r_type in ("rectangular", "rectangle"):
                return RectStartRegion(
                    center=center,
                    width=float(raw.get("width", 40.0)),
                    height=float(raw.get("height", 40.0)),
                )
            raise ValueError(f"Unknown start_region type: {r_type}")

        cfg = NavigationConfig(
            robot_radius_px=data.get("robot_radius_px"),
            ump_rod_width_maze=data.get("ump_rod_width_maze"),
            ump_rod_length_maze=data.get("ump_rod_length_maze"),
            start_region=parse_start_region(data.get("start_region")),
            safety_margin_weight=float(data.get("safety_margin_weight", 5.0)),
            path_resolution=float(data.get("path_resolution", 2.0)),

            path_smoothing_enabled=bool(data.get("path_smoothing_enabled", False)),
            path_smoothing_factor=float(data.get("path_smoothing_factor", 0.3)),
            effective_radius_px_override=data.get("effective_radius_px_override"),
            disable_ump_mask_for_planning=bool(
                data.get("disable_ump_mask_for_planning", False)
            ),
            n_start_samples=int(data.get("n_start_samples", 5)),
            enable_ump_rod_detection=bool(data.get("enable_ump_rod_detection", False)),
            targets_maze={
                k: tuple(v)                           
                for k, v in data.get("targets_maze", {}).items()
            },
            max_arm_switches=int(data.get("max_arm_switches", 1)),
            output_dir=Path(data.get("output_dir", Path("outputs") / "navigation")),
            center_preference_alpha=float(data.get("center_preference_alpha", 2.0)),
            use_exponential_cost=bool(data.get("use_exponential_cost", True)),
        )
        return cfg

def load_config(path: Union[str, Path]) -> NavigationConfig:

    path = Path(path)
    if not path.exists():

        return NavigationConfig()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return NavigationConfig.from_dict(data)

def save_config(config: NavigationConfig, path: Union[str, Path]) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
