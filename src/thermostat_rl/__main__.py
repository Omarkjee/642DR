"""Print the active settings while the simulation and training runner are built."""

from dataclasses import asdict

from thermostat_rl.config import ExperimentConfig


def main() -> None:
    config = ExperimentConfig.from_env()
    print("Active experiment settings (not a training run):")
    for name, value in asdict(config).items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
