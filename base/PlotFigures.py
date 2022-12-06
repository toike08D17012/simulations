import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


class PlotFigures():
    def __init__(
        self,
        num_agents: int,
        save_dir: str or Path,
        fontsize=30,
        suffixes: str or list[str] = ['.jpg', '.pdf', '.eps']
    ):
        """
        Args:
            num_agents(int): The number of agents.
            save_dir(str or pathlib.Path): The path to the save directory.
            fontsize(int): The fontsize in the figure.
            suffixes(str or list[str]): The suffixes for saving figures, or its array.
        """
        self.num_agents = num_agents
        self.save_dir_base = Path(save_dir)
        self.save_dir_base.mkdir(exist_ok=True, parents=True)

        plt.rcParams['font.size'] = fontsize

        self.suffixes = suffixes

    def plot_regret(
        self,
        regrets: list or list[list],
        constraint_violations: list or list[list],
        delays: int or list[int],
        prefixes: str or list[str],
        iteration: None or int = None,
    ):
        """
        Plot regrets and constraint violations.
        Args:
            regrets(list or List[list]): The array of regret or its array.
            constrain_violations(list or List[list]): The array of constraint_violation or its array.
            delays(list or List[list]): regrets, constraint_violationの各値が, 何ステップ遅延の時のものかをあらわす変数orそのリスト
            iteration(int or None): 何ステップ目の情報を描画するか表す変数
        """
        # The type of regrets and constraint_violation convert list[list].
        if not isinstance(regrets[0], list):
            regrets = [regrets]
        if not isinstance(constraint_violations[0], list):
            constraint_violations = [constraint_violations]
        # The type of delays converts list[int]
        if not isinstance(delays, list):
            delays = [delays]
        if not isinstance(prefixes, list):
            prefixes = [prefixes]

        assert len(regrets) == len(constraint_violations) == len(delays),\
            'The length of regrets, constraint_violations, and delays must be the same.'
        for regret, constraint_violation in zip(regrets, constraint_violations):
            assert len(regret) == len(constraint_violation),\
                'The length of regret and constraint_violation in same delay must be the same.'

        # Plot regret.
        save_dir = self.save_dir_base / 'regret'
        save_dir.mkdir(exist_ok=True)
        self._plot_fig(save_dir, regrets, delays, prefixes, iteration, 'regret', 'Reg(T)/T')
        # Plot constraint violation
        save_dir = self.save_dir_base / 'constraint_violation'
        save_dir.mkdir(exist_ok=True)
        self._plot_fig(save_dir, constraint_violations, delays, prefixes, iteration, 'constraint_violation', r'$Reg^c$(T)/T')

    def _plot_fig(self, save_dir, values, delays, prefixes, iteration, save_file_name, y_label, x_label='iteration k'):
        fig = plt.figure(figsize=(16, 9), dpi=60)
        ax = fig.add_subplot()
        for value, delay, prefix in zip(values, delays, prefixes):
            ax.plot(value, label=f'{prefix} feedback, delay: {delay} steps')

        # Added axis labels.
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # Added legends on graph.
        ax.legend()

        # Set option that the legends and axis labels to fit within the graph.
        fig.tight_layout()

        if iteration is None:
            save_path = save_dir / f'{save_file_name}_{self.num_agents}agents_{max(delays)}step_delay'
        else:
            save_path = save_dir / f'{save_file_name}_{self.num_agents}agents_{delay}step_delay_{prefix}_feedback_{iteration // 1000}k'
        self._save_figure(save_path, fig)
        plt.clf()
        plt.close()

    def _save_figure(self, save_path, fig):
        for suffix in self.suffixes:
            fig.savefig(save_path.with_suffix(suffix))

    