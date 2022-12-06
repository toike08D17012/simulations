import argparse
import warnings
from itertools import cycle
from pathlib import Path

from tqdm import tqdm

from base.networks import CreateNetwork
from base.PlotFigures import PlotFigures
from base.utils import fix_seed
from ResourceAllocation.ResorceAllocation import ResourceAllocation

SAVE_DIR = Path('results/resource_allocation/')

AGENT_NUM = 50
SKIP_AGENT_NUM = 20

# 遅延がある場合の最大遅延
DELAY_MAX = 5
# 1step遅延が発生する確率
DELAY_RATE = 5

CONSTRAINT_WEIGHT = 3

GRAPH_NUM = 4

MAX_ITERATION = 5000
FIG_INTERVAL = 5000


def optimize_resource_allocation(args):
    save_dir = args.save_path
    save_dir.mkdir(exist_ok=True, parents=True)

    agent_num = args.agent_num

    graph_num = args.graph_num
    skip_agent_num = args.skip_agent_num

    delay_rate = args.delay_rate

    constraint_weight = args.constraint_weight

    max_iteration = args.max_iteration
    fig_interval = args.fig_interval

    # Fix random seed of numpy.
    fix_seed()

    plotter = PlotFigures(agent_num, save_dir=save_dir / 'figure', suffixes=['.jpg', '.pdf'])

    regret_list = []
    constraint_violation_list = []
    delay_list = []
    prefix_list = []
    for is_bandit in [True, False]:
        if is_bandit:
            prefix = 'bandit'
        else:
            prefix = 'gradient'

        for delay_max in [0, args.delay_max]:
            prefix_list.append(prefix)
            delay_list.append(delay_max)
            network = CreateNetwork(agent_num, delay_max, delay_rate)
            adjs = network.create_adjacency_matrix(graph_num, skip_agent_num, save_dir=save_dir / 'graph')

            algorithm = ResourceAllocation(agent_num, delay_max, is_bandit=is_bandit, constraint_weight=constraint_weight)

            for iteration, adj in enumerate(tqdm(cycle(adjs), total=max_iteration)):
                weight_matrix = network.create_weight_matrix(adj)
                regret, constraint_violation = algorithm(iteration, weight_matrix)

                if (iteration + 1) % fig_interval == 0 or (iteration + 1) == max_iteration:
                    plotter.plot_regret(regret, constraint_violation, delay_max, prefix, iteration + 1)

                if iteration + 1 == max_iteration:
                    break

            regret_list.append(regret)
            constraint_violation_list.append(constraint_violation)

            algorithm.save_data_as_csv(save_dir / 'csv', file_name=f'result_{prefix}_feedback_{delay_max}step_delay.csv')

    plotter.plot_regret(regret_list, constraint_violation_list, delay_list, prefix_list)


def get_args():
    parser = argparse.ArgumentParser(description='リソース配分問題')
    parser.add_argument('-sp', '--save_path', default=SAVE_DIR, type=Path, help='保存先のディレクトリ')
    parser.add_argument('-n', '--agent_num', default=AGENT_NUM, type=int, help='エージェント数')
    parser.add_argument('-gn', '--graph_num', default=GRAPH_NUM, type=int, help='グラフの種類数')
    parser.add_argument('-k', '--skip_agent_num', default=SKIP_AGENT_NUM, type=int, help='グラフを作成する際にとばすエージェントの数')
    parser.add_argument('-dm', '--delay_max', default=DELAY_MAX, type=int, help='遅延がある場合，最大何ステップ遅延するか')
    parser.add_argument('-dr', '--delay_rate', default=DELAY_RATE, type=int, help='1step遅延が発生する確率')
    parser.add_argument('-cw', '--constraint_weight', default=CONSTRAINT_WEIGHT, type=float or int, help='制約の重み')
    parser.add_argument('-mi', '--max_iteration', default=MAX_ITERATION, type=int, help='最大イテレーション')
    parser.add_argument('-fi', '--fig_interval', default=FIG_INTERVAL, type=int, help='図を描画する間隔')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    args = get_args()
    optimize_resource_allocation(args)
