import argparse
import os
import subprocess
import sys
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CORE_SCHEMES = [
    "NoPhysics",
    "PhysicsConcat",
    "PhysicsDiffNoAtt",
    "Full",
]


def _build_command(
    python_exe: str,
    repo_dir: str,
    config_path: str,
    output_dir: str,
    scheme: str,
    seeds: List[int],
    device: str,
) -> List[str]:
    return [
        python_exe,
        "-m",
        "evaluation.run_ablation_experiments",
        "--config",
        config_path,
        "--output-dir",
        output_dir,
        "--ablation-type",
        "Architecture",
        "--scheme",
        scheme,
        "--device",
        device,
        "--seeds",
        *[str(seed) for seed in seeds],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行 AG-S2CNN 核心四变体消融实验")
    parser.add_argument("--config", type=str, default="config_ablation_quick.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/ablation_core4",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(repo_dir, config_path)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("[开始] 运行核心四变体结构消融")
    print(f"[配置] config={config_path}")
    print(f"[输出] output_dir={output_dir}")
    print(f"[设备] device={args.device}")
    print(f"[种子] seeds={args.seeds}")

    for index, scheme in enumerate(CORE_SCHEMES, start=1):
        print(f"[进度] {index}/{len(CORE_SCHEMES)} -> {scheme}", flush=True)
        cmd = _build_command(
            python_exe=args.python,
            repo_dir=repo_dir,
            config_path=config_path,
            output_dir=output_dir,
            scheme=scheme,
            seeds=args.seeds,
            device=args.device,
        )
        subprocess.run(cmd, cwd=repo_dir, check=True)

    print("[完成] 核心四变体消融实验已全部执行完毕")


if __name__ == "__main__":
    main()
