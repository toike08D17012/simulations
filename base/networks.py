import copy
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class CreateNetwork():
    def __init__(
        self,
        num_agents: int,
        delay_max: int,
        delay_rate: int,
        seed: int = 0
    ):
        """
        Args:
            num_agents(int): The number of agents.
            delay_max(int): The maximum steps of time delay.
            delay_rate(int): 遅延1stepの場合の遅延確率(2step以降の場合は"int(delay_rate / i step)"に従って計算)
            seed(int): The seed of random number.
        """
        self.num_agents = num_agents
        self.delay_max = delay_max
        self.num_agents_expand = num_agents * (delay_max + 1)

        self.delay_separater = np.cumsum([1] + [int(delay_rate / (i + 1)) for i in range(self.delay_max)])

    def create_adjacency_matrix(
        self,
        graph_num: int,
        skip_agent_num: int,
        graph_type: str = 'uniformly_strong_connected',
        save_graph: bool = True,
        save_dir: str or Path or None = None,
    ) -> list[np.ndarray]:
        """
        Create the adjacency matrices.
        Args:
            graph_num(int): The number of graphs to be used.
            skip_agent_num(int): グラフを作成するときに飛ばすエージェントの数．
            graph_type(str): グラフの種類. 一様強連結 or 完全グラフのみ利用可能
            save_graph(bool): The flag which indicates if the graphs to be saved or not.
            save_dir(str or Path or None): The path to the directory to be saved.
        Returns:
            adjs(List[numpy.ndarray]): The array of adjacency matrix.
        """
        adjs = self._init_graph(self.num_agents, skip_agent_num, graph_num, graph_type)

        if save_graph:
            assert save_dir is not None, 'When the graph is saved, save_dir must be set.'
            self._save_graph(Path(save_dir), copy.deepcopy(adjs))
        return adjs

    def _init_graph(self, num_agents: int, skip_agent_num: int, graph_num: int, graph_type: str) -> list[np.ndarray]:
        assert graph_num <= 4, 'The number of graphs must be set less than 4.'
        assert num_agents > skip_agent_num, 'skip_agent_num must be set less than num_agents.'

        if graph_type == 'complete_graph':
            adjs = [np.ones((num_agents, num_agents))]
        elif graph_type == 'uniformly_strong_connected':
            adjs = []
            # 1つ目のグラフを作成
            tmp_adj = np.zeros((num_agents, num_agents))
            for i in range(num_agents):
                for j in range(num_agents):
                    if (
                        i + skip_agent_num == j
                        or (j == 1 and i == num_agents - skip_agent_num + 1)
                        or (j == 0 and i == num_agents - skip_agent_num)
                        or i == j
                    ):
                        tmp_adj[i, j] = 1
            for i in range(skip_agent_num):
                tmp_adj[num_agents - skip_agent_num + i, i] = 1
            adjs.append(copy.deepcopy(tmp_adj))
            # 2つ目は1つ目の転置を入れる
            adjs.append(copy.deepcopy(tmp_adj.T))

            # 3つ目のグラフを作成
            tmp_adj = np.zeros((num_agents, num_agents))
            for i in range(num_agents):
                for j in range(num_agents):
                    if i + 1 == j or (i == num_agents - 1 and j == 0) or i == j:
                        tmp_adj[i, j] = 1
            adjs.append(copy.deepcopy(tmp_adj))
            # 4つ目は3つ目の転置を入れる
            adjs.append(copy.deepcopy(tmp_adj.T))
        else:
            raise ValueError('graph_type is must be set "complete_graph" or "uniformly_strongly_connected".')

        while len(adjs) > graph_num:
            adjs.pop()

        return adjs

    def _save_graph(self, save_dir, adjs, figsize=(11, 11), dpi=60):
        save_dir.mkdir(exists_ok=True, parents=True)
        for idx, adj in enumerate(adjs):
            # selfループを描画しないように隣接行列を修正
            for j in range(self.num_agents):
                adj[j, j] = 0

            g = nx.from_numpy_matrix(adj.T, create_using=nx.MultiDiGraph())
            pos = nx.circular_layout(g)

            plt.figure(figsize=figsize, dpi=dpi)
            labels = {}
            for i in range(self.num_agents):
                labels[i] = str(i + 1)

            # Draw nodes.
            nx.draw_networkx_nodes(g, pos, node_size=4800, alpha=1.0, node_color='lightblue', margins=0)
            # Draw edges.
            nx.draw_networkx_edges(g, pos, width=5, arrowsize=125)
            # Draw node labels.
            nx.draw_networkx_labels(g, pos, labels, font_size=60)

            plt.axis('off')
            plt.tight_layout()
            save_path = save_dir / f'network_{self.num_agents}_{idx}.jpg'
            plt.savefig(save_path)
            plt.savefig(save_path.with_suffix('.pdf'))
            plt.savefig(save_path.with_suffix('.eps'))

            plt.clf()
            plt.close()

    def create_weight_graph(self, adj: np.ndarray):
        expand_matrix = self._create_expand_graph(adj)
        weight_matrix = self._create_weight_matrix(expand_matrix)

        return weight_matrix

    def _create_expand_graph(self, adj):
        # 何ステップ遅延するか決定
        tau = np.random.randint(0, 100, (self.num_agents, self.num_agents))
        delay_times = range(1, self.delay_max)
        # 遅延確率の調整
        for delay_time in delay_times:
            tau = np.where(
                (self.delay_separater[delay_time - 1] <= tau) & (tau < self.delay_separater[delay_time]),
                delay_time,
                tau
            )

        tau = np.where(tau >= self.delay_max, 0, tau)
        np.fill_diagonal(tau, 0)

        # Based on the delayed conditions, create expand graph.
        expand_adj = np.zeros((self.num_agents_expand, self.num_agents_expand))
        expand_adj[:self.num_agents, :self.num_agents] = np.where(tau == 0, adj, expand_adj[:self.num_agents, :self.num_agents])

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if tau[i, j] != 0:
                    expand_adj[i + (tau[i][j] - 1) * self.num_agents][j] = adj[i][j]

        for i in range(self.num_agents_expand - self.num_agents):
            expand_adj[i, i + self.num_agents] = 1

        return expand_adj

    def _create_weight_matrix(self, expand_graph):
        weight_matrix = expand_graph / np.sum(expand_graph, axis=0)
        return weight_matrix
