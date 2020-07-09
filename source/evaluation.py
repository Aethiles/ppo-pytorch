import json
import matplotlib
import numpy as np
import operator
import os
import sys

from collections import deque
from dataclasses import dataclass, field
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import List, Tuple, Union
from matplotlib.lines import Line2D


@dataclass
class Run:
    steps: np.ndarray
    values: np.ndarray

    def get_dict(self,
                 ):
        return {'steps': self.steps.tolist(), 'values': self.values.tolist()}


@dataclass
class Experiment:
    path: str
    game: str
    runs: List[Run] = field(default_factory=lambda: list())
    baseline: Run = None

    def get_dict(self,
                 ):
        return self.__dict__

    def save(self,
             ):
        """
        Converts the experiment to json and saves it to the given path
        :param path:
        :return:
        """
        path = os.path.join(self.path, self.game) + '.json'
        with open(path, 'w') as f:
            f.write(self.to_json())

    def to_json(self):
        """
        Converts self to json
        :return:
        """
        return json.dumps(self, default=lambda o: o.get_dict(), sort_keys=True)

    def score(self):
        path = os.path.join(self.path, 'final_score')
        with open(path, 'a') as f:
            f.write(f'{self.game}:\n')
            for run in self.runs:
                score = np.mean(run.values[-100:])
                f.write(f'\t{score}\n')
            f.write(f'\n')

    def smooth(self,
               window_size: int = 16,
               ):
        """
        Smooths the data by computing an average on the last window_size episode scores
        :param window_size:
        :return:
        """
        for run in self.runs:
            smooth_data = []
            buffer = deque(maxlen=window_size)
            for value in run.values:
                buffer.append(value)
                smooth_data.append(np.mean(buffer))
            run.values = smooth_data[window_size//2:]
            run.steps = run.steps[:-window_size//2]


def setup():
    """
    Sets figure size and grid of the plot.
    :return: None
    """
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams['font.size'] = 20
    plt.grid(alpha=0.25)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)


def create_plot(experiment: Experiment,
                comparison: Tuple,
                default_trend: Tuple = None,
                plot: bool = False,
                ):
    """

    :param experiment:
    :param plot:
    :return:
    """
    colors = ['#307B3B', '#CAA023', '#254796', '#F1009A']
    setup()
    for i, run in enumerate(experiment.runs):
        plt.scatter(run.steps, run.values, label=f'Run {i+1}', color=colors[i], s=18)
    plt.plot(comparison[0], comparison[1], color='black', label='Trendline')
    plt.title(f'{experiment.game}')
    plt.xlabel('Time step')
    plt.ylabel('Episode score')
    make_legend(colors, trend=False)
    plt.tight_layout(pad=1.02)

    if plot:
        plt.show()
    else:
        path = os.path.join(experiment.path, experiment.game)
        plt.savefig(path)
        if default_trend is not None:
            plt.plot(default_trend[0], default_trend[1], label='Trendline (reference)', color=colors[3])
            make_legend(colors)
            plt.tight_layout(pad=1.02)
            plt.savefig(path + '_reference')
    plt.clf()


def make_legend(colors: List,
                trend: bool = True):
    legend_elements = [Line2D([0], [0], color='black', lw=4, label='Trendline'),
                       Line2D([0], [0], marker='o', color='w', label='Run 1', markerfacecolor=colors[0], markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Run 2', markerfacecolor=colors[1], markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Run 3', markerfacecolor=colors[2], markersize=15)]
    if trend:
        legend_elements.insert(1, Line2D([0], [0], color=colors[3], lw=4, label='Trendline (reference)'))
    plt.gca().legend(handles=legend_elements)

def create_trendline(data: Experiment,
                     window: int,
                     ):
    data_x = np.empty(0)
    data_y = np.empty(0)
    for run in data.runs:
        data_x = np.concatenate((data_x, run.steps))
        data_y = np.concatenate((data_y, run.values))
    values = deque(maxlen=window)
    data_x, data_y = zip(*sorted(zip(data_x, data_y), key=operator.itemgetter(0)))
    trend_y = []
    for y in data_y:
        values.append(y)
        trend_y.append(np.mean(values))
    return data_x[:-window//2], trend_y[window//2:]
        

def parse(directory: str,
          experiment: str = None,
          ):
    """

    :param directory:
    :return:
    """
    if experiment is not None and '16_envs' in experiment:
        step_scale = 16
    elif experiment is not None and '4_envs' in experiment:
        step_scale = 4
    else:
        step_scale = 8
    acc = EventAccumulator(directory)
    acc.Reload()
    _, steps, values = zip(*acc.Scalars('Episode/avg_reward'))
    config = str(acc.Tensors('Config/text_summary')[0][2].string_val)[5:-2].replace('\\n', '\n')
    return np.array(steps) * step_scale, np.array(values), config


def create_default_config_trends(data_dir: str):
    runs = os.listdir(data_dir)
    data = {'BeamRider': Experiment('', 'BeamRider'),
            'Breakout': Experiment('', 'Breakout'),
            'Pong': Experiment('', 'Pong'),
            'Seaquest': Experiment('', 'Seaquest'),
            'SpaceInvaders': Experiment('', 'SpaceInvaders')}
    for run in runs:
        steps, values, _ = parse(os.path.join(data_dir, run))
        game = str(run.split('-')[0])
        data[game].runs.append(Run(steps, values))

    trends = {'BeamRider': (),
              'Breakout': (),
              'Pong': (),
              'Seaquest': (),
              'SpaceInvaders': ()}
    for d in data:
        trends[d] = create_trendline(data[d], 256)
    return trends


def main(root_dir: str,
         default_config_dir: str = None,
         ):
    """

    :param root_dir:
    :return:
    """
    trends = None
    if default_config_dir is not None:
        trends = create_default_config_trends(default_config_dir)
    experiments = os.listdir(root_dir)
    print(f'Full list of experiments: {experiments}')
    for experiment in experiments:
        print(f'Evaluating {experiment}')
        path = os.path.join(root_dir, experiment)
        data = {'BeamRider': Experiment(path, 'BeamRider'),
                'Breakout': Experiment(path, 'Breakout'),
                'Pong': Experiment(path, 'Pong'),
                'Seaquest': Experiment(path, 'Seaquest'),
                'SpaceInvaders': Experiment(path, 'SpaceInvaders')}
        runs = sorted(os.listdir(path))
        for run in runs:
            steps, values, config = parse(os.path.join(path, run), experiment)
            game = str(run.split('-')[0])
            data[game].runs.append(Run(steps, values))

        with open(os.path.join(path, 'config.md'), 'w') as f:
            f.write(config)

        for d in data:
            data[d].save()
            data[d].score()
            comparison = create_trendline(data[d], 256)
            data[d].smooth()
            if trends is not None:
                create_plot(data[d], comparison, trends[d])
            else:
                create_plot(data[d], comparison)


if __name__ == '__main__':
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        main()

