from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
from rich.columns import Columns

from collections import defaultdict
import time
import re


from ..core.config import Config
from ..core.metric_tracker import MetricTracker




class Visualiser:

    def __init__(self, config: Config):

        self.config = config
        self.past_count = 90
        self.recent_history = []

        self.console = Console()
        self.console.clear()
        self.live = Live(None, console=self.console, auto_refresh=False, redirect_stdout=True)
        self.live.start()

        self.timings = defaultdict(list)
        self.curr_status = None
        self.current_timing = time.perf_counter()
        
    def update_step(self, timestep: int, batch_run: int, total_batch_runs: int, status: str, tracker: MetricTracker, checkpoint: bool = False):
        layout = self.generate_dashboard_layout(timestep, batch_run, total_batch_runs, status, tracker)
        self.live.update(layout, refresh=True)

        now = time.perf_counter()
        if self.curr_status:
            self.timings[self.curr_status].append(round(now - self.current_timing, 2))
            self.timings[self.curr_status] = self.timings[self.curr_status][-self.past_count:]
        self.curr_status = status
        self.current_timing = now

        if checkpoint:
            current_averages = {key: tracker.get_average(key) for key in tracker.metrics.keys()}
            if current_averages:
                self.recent_history.append(current_averages)
                self.recent_history = self.recent_history[-self.past_count:]

    def generate_config_table(self):

        def make_panel(title, params):
            table = Table(show_header=False, box=None)
            table.add_column("Param", style="dim")
            table.add_column("Value", style="bold cyan", justify="right")
            for label, value in params:
                table.add_row(label, str(value))

            return Panel(table, title=f"[bold]{title}[/]", border_style="white")

        training_params = [
            ("Train Timesteps", f"{self.config.training.training_timesteps}"),
            ("Sim Timesteps", f"{self.config.training.simulation_timesteps}"),
            ("Timestep", f"{self.config.training.timestep}"),
            ("Parallel Worlds", f"{self.config.training.worlds_parallised}"),
            ("Seed", f"{self.config.training.seed}"),
            ("Save Interval", f"{self.config.training.periodic_save_interval}"),
        ]
        
        ppo_params = [
            ("PPO Epochs", f"{self.config.mappo.ppo_epochs}"),
            ("Gamma", f"{self.config.mappo.gamma}"),
            ("GAE Lambda", f"{self.config.mappo.gae_lambda}"),
            ("Clip Coef", f"{self.config.mappo.clip_coef}"),
            ("VF Coef", f"{self.config.mappo.vf_coef}"),
            ("Entropy Coef", f"{self.config.mappo.ent_coef}"),
            ("Actor LR", f"{self.config.mappo.actor_learning_rate:.1e}"),
            ("Critic LR", f"{self.config.mappo.critic_learning_rate:.1e}"),
        ]

        arch_params = [
            ("LSTM Hidden", f"{self.config.actor.lstm_hidden_size}"),
            ("Feature Dim", f"{self.config.actor.feature_dim}"),
        ]

        lang_params = [
            ("Vocab Size", f"{self.config.comms.vocab_size}"),
            ("Comm Size", f"{self.config.comms.communication_size}"),
            ("Num Comms", f"{self.config.comms.num_comms}"),
        ]

        aim_params = [
            ("HQ Layers", f"{self.config.comms.rq_levels}"),
            ("AIM LR", f"{self.config.aim_training.aim_learning_rate:.1e}"),
            ("Obs Runs", f"{self.config.aim_training.obs_runs}"),
            ("Obs Noise", f"{self.config.aim_training.obs_runs_noise}"),
            ("AIM Batch Size", f"{self.config.aim_training.aim_batch_size}"),
        ]

        return Columns([
            make_panel("Training", training_params),
            make_panel("PPO Hyperparameters", ppo_params),
            make_panel("Architecture", arch_params),
            make_panel("Language", lang_params),
            make_panel("Create AIM Language", aim_params)
        ])
        
    def generate_dashboard_layout(self, timestep: int, batch_run: int, total_batch_runs: int, status: int, tracker: MetricTracker = None):
        
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
        )
        progress.add_task("Batch Progress", completed=int(batch_run), total=int(total_batch_runs))

        metrics_table = Table(show_header=True, header_style="bold magenta", expand=True, box=None)
        metrics_table.add_column("Metric", style="dim")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column(f"Trend (Past {self.past_count} steps)", justify="left")

        colours = ["red", "green", "blue", "yellow", "magenta", "cyan", "white"]

        to_readable_from_camel = lambda s: re.sub(r"(?<!^)(?=[A-Z])", " ", s).title()
        if tracker:
            for key in tracker.metrics.keys():
                label = to_readable_from_camel(key)
                colour = colours[ord(key[0]) % len(colours)]
                last_entry = self.recent_history[-1] if self.recent_history else {}
                val = last_entry.get(key, "0")
                
                try:
                    display_val = f"{float(val):.4f}"
                except (ValueError, TypeError):
                    display_val = str(val)
                
                line = get_line_graph(self.recent_history, key) if self.recent_history else ""
                metrics_table.add_row(label, display_val, f"[{colour}]{line}[/]")

        timing_texts = []
        for key, vals in self.timings.items():
            timing_texts.append(f"{to_readable_from_camel(key)} ({sum(vals) / len(vals):.2f}(s))")
        timing_text = " | ".join(timing_texts)


        monitor_group = Group(
            Align.right(Text(timing_text, style="dim")),
            f"[bold yellow]Status:[/] {status}",
            " ",
            progress,
            " ",
            metrics_table
        )
        
        monitor_panel = Panel(
            monitor_group, 
            title=f"[bold]Training - Comm Type is {self.config.comms.communication_type.upper()} - Step {timestep}[/]", 
            border_style="blue"
        )

        config_panel = self.generate_config_table()

        return Columns([config_panel, monitor_panel])


    def stop(self):
        self.live.stop()


def get_line_graph(data, key):
    
    y = [float(row[key]) for row in data]

    if not y:
        return ""

    chars = " ▂▃▄▅▆▇█"
    
    low, high = min(y), max(y)
    if high == low:
        return chars[0] * len(y)
        
    line = ""
    for v in y:
        idx = int((v - low) / (high - low) * (len(chars) - 1))
        line += chars[idx]
    return line
