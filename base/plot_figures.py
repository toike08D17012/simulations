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
        self.save_dir_base.mkdir(exists_ok=True, parents=True)

        plt.rcParams['font.size'] = fontsize

        self.suffixes = suffixes

    def plot_regret(
        self,
        regrets: list or list[list],
        constraint_violations: list or list[list],
        delays: list or list[int],
        iteration: None or int = None,
    ):
        """
        Plot regrets and constraint violations.
        Args:
            regrets(list or List[list]): The array of regret or its array.
            constrain_violation(list or List[list]): The array of constraint_violation or its array.
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

        assert len(regrets) == len(constraint_violations) == len(delays),\
            'The length of regrets, constraint_violations, and delays must be the same.'
        for regret, constraint_violation, delay in zip(regrets, constraint_violations, delays):
            assert len(regret) == len(constraint_violation),\
                'The length of regret and constraint_violation in same delay must be the same.'

        save_dir = self.save_dir_base / 'regret'
        # Plot regret.
        self._plot_fig(save_dir, regret, delays, iteration, 'regret', 'Reg(T)/T')
        # Plot constraint violation
        self._plot_fig(save_dir, constraint_violations, delays, iteration, 'constraint_violation', r'$Reg^c$(T)/T')

    def _plot_fig(self, save_dir, values, delays, iteration, save_file_name, y_label, x_label='iteration k'):
        fig = plt.figure(figsize=(16, 9), dpi=60)
        ax = fig.add_subplot()
        for value, delay in zip(values, delays):
            ax.plot(value, label=f'delay: {delay} steps')

        # Added axis labels.
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # Added legends on graph.
        ax.legend()

        # Set option that the legends and axis labels to fit within the graph.
        fig.tight_layout()

        if iteration is None:
            save_path = save_dir / f'{save_file_name}_{self.num_agents}agents_{delay}step_delay'
        else:
            save_path = save_dir / f'{save_file_name}_{self.num_agents}agents_{delay}step_delay_{iteration // 1000}k'
        self._save_fig(save_path, fig)

    def _save_figure(self, save_path, fig):
        for suffix in self.suffixes:
            fig.savefig(save_path.with_suffix(suffix))
