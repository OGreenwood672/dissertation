import time
import os
import csv
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
from rich.columns import Columns
from collections import defaultdict
import re

from controller.marl.config import CommunicationType, Config
from logging_utils.decorators import LoggingFunctionIdentification

class MetricTracker:
    def __init__(self):
        self.reset_tracker()
    
    def update(self, key, value):
        self.metrics[key] += value
        self.counts[key] += 1
    
    def get_average(self, key, default=""):
        if key not in self.metrics:
            return default
        
        return self.metrics[key] / self.counts[key]

    def reset_tracker(self):
        self.metrics = defaultdict(int)
        self.counts = defaultdict(int)

class Logging:
    def __init__(self, save_folder: str, config: Config, start_step: int = 0, num_codebooks: int = 1, mode: str = "RL"):
        
        self.config = config
        
        assert os.path.exists(save_folder), f"Save folder {save_folder} does not exist"
        self.save_file = os.path.join(save_folder, "log.csv")

        if mode == "RL":
            
            self.headers = [
                "timestep",
                "reward_mean", "reward_std",
                "actor_loss", "critic_loss",
                "entropy_loss", "action_loss", "comm_loss",
                "lstm_grad_norm", "comm_grad_norm", "action_grad_norm",
                "comm_entropy", "comm_perplexity",
                "predicted_return_loss", "predicted_sender_goal_loss", "predicted_receiver_goal_loss"
            ]
            self.headers += [s for x in range(num_codebooks) for s in [f"usage_count_{x}", f"std_usage_{x}", f"topk_usage_{x}"]]
        
        elif mode == "language":
            self.headers = [
                "timestep",
                "train_commitment_loss", "train_reconstruction_loss",
                "validation_commitment_loss", "validation_reconstruction_loss"
            ]
        
        elif mode == "imitate":
            self.headers = [
                "timestep",
                "train_actor_loss", "train_critic_loss", "train_accuracy",
                "validation_actor_loss", "validation_critic_loss", "validation_accuracy",
                "test_actor_loss", "test_critic_loss", "test_accuracy",
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")


        file_exists = os.path.exists(self.save_file)

        if file_exists:
            self.clear_logs(start_step)

        self.file = open(self.save_file, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)

        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    @LoggingFunctionIdentification("LOGGER")
    def log(self, timestep: int, tracker: MetricTracker):
        
        datarow = {h: tracker.get_average(h, default="") for h in self.headers}
        datarow["timestep"] = timestep
        
        try:
            self.writer.writerow(datarow)
            self.file.flush()
        except ValueError as e:
            print(f"Failed to log row: {e}")
    
    
    def clear_logs(self, start_step):

        if input(f"Logs must be truncated after time step {start_step}. Do you wish to proceed (Y/n)").lower() == "n":
            quit()

        print(f"Truncating log at timestep {start_step} and onwards...")

        try:
            rows = []
            with open(self.save_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if int(float(row["timestep"])) < start_step:
                            rows.append(row)
                    except ValueError:
                        continue

            with open(self.save_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()

        except Exception as e:
            print(f"Error clearing logs: {e}")

    def close(self):
        self.file.close()


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
            ("Buffer Size", f"{self.config.training.buffer_size}"),
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

class ObsLogger:

    def __init__(self, save_file):

        self.save_file = save_file

        file_exists = os.path.exists(self.save_file)

        if file_exists:
            os.remove(self.save_file)

        self.file = open(self.save_file, "a", newline="")
        self.writer = csv.writer(self.file)

    @LoggingFunctionIdentification("LOGGER")
    def log(self, obs):
        try:
            self.writer.writerows(obs)
        except ValueError as e:
            print(f"Failed to log row: {e}")


    def close(self):
        self.file.close()

class CommsLogger:

    def __init__(self, save_folder, vocab_size):

        self.save_folder = save_folder
        self.vocab_size = vocab_size

        os.makedirs(save_folder, exist_ok=True)

        for file in os.listdir(save_folder):
            os.remove(os.path.join(save_folder, file))

        self.save_files = [open(os.path.join(save_folder, f"comms_{i}.csv"), "a", newline="") for i in range(vocab_size)]

        self.writers = [csv.writer(file) for file in self.save_files]


    @LoggingFunctionIdentification("LOGGER")
    def log(self, comm_index, obs):
        try:
            self.writers[comm_index].writerow(obs)
        except ValueError as e:
            print(f"Failed to log row: {e}")

    def close(self):
        for writer in self.writers:
            writer.close()
    
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
