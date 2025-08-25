import argparse
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from igafsa_funcs.grideaf_foodconsistence import GrideAF_foodconsistence
from igafsa_funcs.arry2orxy import arry2orxy
from igafsa_funcs.gridaf_prey import GridAF_prey
from igafsa_funcs.gridaf_swarm import GridAF_swarm
from igafsa_funcs.gridaf_follow import GridAF_follow
from igafsa_funcs.distance import distance
from igafsa_funcs.smooth import Smooth
from igafsa_funcs.bezier_smooth import bezierSmooth


@dataclass
class IGAFSAConfig:
    """
    运行配置（保持默认参数与原脚本一致，尽量不改变行为）
    """
    env_path: str = "object3_2.txt"
    N: int = 50
    try_number: int = 8
    MAXGEN: int = 100
    visual: int = 10
    delta: float = 0.618
    start_index: int = 0
    goal_index: Optional[int] = None  # 若为 None，则默认使用 n*n-1
    shift: int = 1
    shiftFreq: int = 4
    rightInf: float = float(np.sqrt(2))
    plot: bool = True
    seed: Optional[int] = 42  # 设置随机种子以保证可复现（None 表示不设种子）

    def resolve_goal(self, n: int) -> int:
        return int(self.goal_index) if self.goal_index is not None else int(n * n - 1)

    def validate(self, a: np.ndarray, n: int, start: int, goal: int) -> None:
        if not (0 <= start < n * n) or not (0 <= goal < n * n):
            raise ValueError("start 或 goal 超出地图范围")
        # 0 表示障碍（根据原脚本 Barrier = np.where(a.ravel() == 0)[0]）
        flat = a.ravel()
        if flat[start] == 0:
            raise ValueError("起点位于障碍物上")
        if flat[goal] == 0:
            raise ValueError("终点位于障碍物上")


class IGAFSARunner:
    """
    将原脚本主流程封装为运行器，尽量不改变算法细节，便于后续扩展、测试与复用。
    """
    def __init__(self, cfg: IGAFSAConfig):
        self.cfg = cfg

    def run(self) -> Dict[str, Any]:
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

        start_time = time.time()

        # 加载障碍环境
        a = np.loadtxt(self.cfg.env_path)
        n = a.shape[0]
        b = np.copy(a)

        # 可选绘图：环境
        if self.cfg.plot:
            plt.figure()
            plt.imshow(b, cmap='gray', origin='lower', extent=[0, n, 0, n])
            plt.xticks(np.arange(0, n - 1, 10))
            plt.yticks(np.arange(0, n - 1, 10))
            plt.text(-3, -1, 'START', color='red', fontsize=10)
            plt.text(n, n, 'GOAL', color='red', fontsize=10)

        # 障碍物矩阵（0 为障碍）
        Barrier = np.where(a.ravel() == 0)[0]

        # 人工鱼群参数（保持原逻辑）
        N = self.cfg.N
        try_number = self.cfg.try_number
        MAXGEN = self.cfg.MAXGEN
        visual = self.cfg.visual
        delta = self.cfg.delta
        start_arr = np.array([self.cfg.start_index])
        goal_arr = np.array([self.cfg.resolve_goal(n)])
        shift = self.cfg.shift
        shiftFreq = self.cfg.shiftFreq
        rightInf = self.cfg.rightInf

        # 运行前校验
        self.cfg.validate(a, n, int(start_arr.item()), int(goal_arr.item()))

        # 初始化群体
        ppValue = np.full((1, N), start_arr)  # 与原脚本保持一致的形态
        position = np.tile(ppValue, (MAXGEN, 1))
        H = GrideAF_foodconsistence(n, ppValue, goal_arr)
        BH, BHindex = np.min(H), np.argmin(H)
        BX = ppValue[:, BHindex]
        count = 1
        runDist = 0.0
        runDist_part = 0.0
        BestH = np.zeros((1, MAXGEN))
        reach_index = []

        # 主循环
        for gen in range(MAXGEN):
            if shift == 1:
                for i in range(N):
                    if gen == 0:
                        nextPosition, nextPositionH = GridAF_prey(
                            n, N, ppValue[:, i], position[gen:gen + 1], i, try_number, H, Barrier,
                            goal_arr, gen, MAXGEN, rightInf, BX, a
                        )
                    else:
                        nextPosition, nextPositionH = GridAF_prey(
                            n, N, ppValue[:, i], position[gen - 1:gen], i, try_number, H, Barrier,
                            goal_arr, gen, MAXGEN, rightInf, BX, a
                        )
                    position[gen][i] = int(np.asarray(nextPosition).ravel()[0])
                    H[i] = float(np.asarray(nextPositionH).ravel()[0])
            else:
                for i in range(N):
                    nextPosition_S, nextPositionH_S = GridAF_swarm(
                        n, N, position[gen - 1:gen], i, visual, delta, try_number, H,
                        Barrier, goal_arr, gen, MAXGEN, rightInf, BX, a
                    )
                    nextPosition_F, nextPositionH_F = GridAF_follow(
                        n, N, position[gen - 1:gen], i, visual, delta, try_number, H,
                        Barrier, goal_arr, gen, MAXGEN, rightInf, BX, a
                    )
                    if nextPositionH_F < nextPositionH_S:
                        nextPosition = nextPosition_F
                        nextPositionH = nextPositionH_F
                    else:
                        nextPosition = nextPosition_S
                        nextPositionH = nextPositionH_S
                    position[gen][i] = int(np.asarray(nextPosition).ravel()[0])
                    H[i] = float(np.asarray(nextPositionH).ravel()[0])

            count += 1
            shift = 2 if count % shiftFreq == 0 else 1
            ppValue = position[gen:gen + 1]
            BH, BHindex = np.min(H), np.argmin(H)
            BX = ppValue[:, BHindex]
            reach_index = np.where(position[gen, :] == (goal_arr[0] - 1))[0]
            if reach_index.size > 0:
                break

        result: Dict[str, Any] = {
            "reached": False,
            "distance_igafsa": None,
            "distance_smooth": None,
            "distance_bezier": None,
            "path_indices": None,
            "path_xy": None,
            "smooth_path": None,
            "bezier_path": None,
        }

        if reach_index.size == 0 and gen >= MAXGEN - 1:
            print('没有路径可以到达目标!!!')
        else:
            transimit = np.array([])
            for i in reach_index:
                arrayValue = position[:, i]
                arrayValue = np.trim_zeros(arrayValue, 'b')
                arrayValue = np.concatenate((start_arr, arrayValue, goal_arr))  # 拼接为一维数组
                arrayValue = arrayValue.reshape(1, -1)  # 一行
                for k in range(arrayValue.shape[1] - 1):
                    d = distance(n, int(arrayValue[0, k]), int(arrayValue[0, k + 1]))
                    runDist_part += d
                transimit = np.append(transimit, runDist_part)
                runDist_part = 0.0

            runDist, runMin_index = np.min(transimit), np.argmin(transimit)
            arrayValue = position[:, reach_index[runMin_index]]
            arrayValue = np.trim_zeros(arrayValue, 'b')
            arrayValue = np.concatenate((start_arr, arrayValue, goal_arr))
            arrayValue = arrayValue.reshape(1, -1)
            print(f'IGAFSA行走长度为: {runDist}')

            for i in range(arrayValue.shape[1]):
                BestH[0, i] = float(int(goal_arr.item()) - int(arrayValue[0, i]))

            row, col = np.unravel_index(arrayValue.astype(int), (n, n))
            array_x, array_y = arry2orxy(n, row, col)

            if self.cfg.plot:
                plt.plot(array_x[0, :] + 0.5, array_y[0, :] + 0.5, 'r', linewidth=2, marker='o')

            Optimal_path = np.column_stack((array_x.ravel(), array_y.ravel()))
            _, idx = np.unique(Optimal_path, axis=0, return_index=True)
            idx_sorted = np.sort(idx)
            Optimal_path = Optimal_path[idx_sorted]
            # 转换为 n*4（正序）
            Optimal_path = np.flipud(Optimal_path)
            Optimal_path = np.hstack([Optimal_path, np.zeros((Optimal_path.shape[0], 2))])
            Optimal_path[:-1, 2] = Optimal_path[1:, 0]
            Optimal_path[:-1, 3] = Optimal_path[1:, 1]
            Optimal_path = np.delete(Optimal_path, -1, axis=0)

            # 平滑
            Path, distanceX = Smooth(a, Optimal_path, 0.75)
            print(f'IGAFSA+Smooth行走长度为: {distanceX}')

            # 贝塞尔平滑
            newPath, distanceB = bezierSmooth(Path)
            if self.cfg.plot:
                plt.plot(newPath[:, 0] + 0.5, newPath[:, 1] + 0.5, 'm', linewidth=2)
                plt.show()
            print(f'IGAFSA+Smooth+bezier行走长度为: {distanceB}')

            result.update({
                "reached": True,
                "distance_igafsa": float(runDist),
                "distance_smooth": float(distanceX),
                "distance_bezier": float(distanceB),
                "path_indices": arrayValue.astype(int),
                "path_xy": np.column_stack((array_x.ravel(), array_y.ravel())),
                "smooth_path": Path,
                "bezier_path": newPath,
            })

        end_time = time.time()
        print(f'代码运行时间： {end_time - start_time:.2f} 秒')
        result["elapsed_seconds"] = end_time - start_time
        return result


def IAFSA():
    """
    兼容原入口：使用默认配置直接运行。
    """
    cfg = IGAFSAConfig()
    runner = IGAFSARunner(cfg)
    return runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGAFSA Runner")
    parser.add_argument("--env", type=str, default="object3_2.txt", help="障碍环境文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    parser.add_argument("--plot", type=int, default=1, help="是否绘图，1=绘图，0=不绘图")
    parser.add_argument("--goal", type=int, default=None, help="终点索引（默认 n*n-1）")
    parser.add_argument("--start", type=int, default=0, help="起点索引（默认 0）")
    args = parser.parse_args()

    cfg = IGAFSAConfig(
        env_path=args.env,
        seed=args.seed,
        plot=bool(args.plot),
        goal_index=args.goal,
        start_index=args.start,
    )
    runner = IGAFSARunner(cfg)
    runner.run()
