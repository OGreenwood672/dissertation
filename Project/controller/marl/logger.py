import os
import csv
from rich.table import Table
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
from rich.columns import Columns

from logging_utils.decorators import LoggingFunctionIdentification


class Logger:

    def __init__(self, save_folder, config, start_step=0):

        self.config = config
        self.recent_history = []
        
        assert os.path.exists(save_folder), f"Save folder {save_folder} does not exist"
        self.save_file = os.path.join(save_folder, "log.csv")

        self.headers = [
            "timestep",
            "reward_mean", "reward_std",
            "communication_entropy_mean", "communication_perplexity_mean",
            "codebook_loss", "communication_active_codebook_usage",
            "actor_loss", "critic_loss", "entropy_loss", "vq_loss"
        ]

        file_exists = os.path.exists(self.save_file)

        if file_exists:
            self.clear_logs(start_step)

        self.file = open(self.save_file, "a")
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)

        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

        self.console = Console()
        self.live = Live(None, console=self.console, auto_refresh=False, redirect_stdout=True)
        self.live.start()
        
    @LoggingFunctionIdentification("LOGGER")
    def log(self, **kwargs):
        
        datarow = {h: kwargs.get(h, "") for h in self.headers}
        
        try:
            self.writer.writerow(datarow)
            self.file.flush()
        except ValueError as e:
            print(f"Failed to log row: {e}")

        self.recent_history.append(datarow)
        self.recent_history = self.recent_history[-30:]

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

    def update_step(self, timestep, batch_run, total_batch_runs, status):
        layout = self.generate_dashboard_layout(timestep, batch_run, total_batch_runs, status)
        self.live.update(layout, refresh=True)

    def generate_config_table(self):

        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Param", style="dim")
        table.add_column("Value", style="bold cyan", justify="right")

        params = [
            ("Actor LR", f"{self.config.actor_learning_rate:.1e}"),
            ("Critic LR", f"{self.config.critic_learning_rate:.1e}"),
            ("Entropy Coef", f"{self.config.ent_coef}"),
            ("Clip Coef", f"{self.config.clip_coef}"),
            ("Batch Size", f"{self.config.buffer_size}"),
            ("PPO Epochs", f"{self.config.ppo_epochs}"),
            ("Parallel Worlds", f"{self.config.worlds_parallised}"),
            ("Comm Type", f"{self.config.communication_type.name}"),
            ("Comm Size", f"{self.config.communication_size}"),
            ("Seed", f"{self.config.seed}"),
        ]

        for label, value in params:
            table.add_row(label, value)
            
        return Panel(table, title="[bold]Hyperparameters[/]", border_style="white")
        
    def generate_dashboard_layout(self, timestep, batch_run, total_batch_runs, status):
        
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
        metrics_table.add_column("Trend (30 steps)", justify="left")

        tracked_metrics = [
            ("Mean Reward", "reward_mean", "green"),
            ("Std Reward", "reward_std", "green"),
            ("Entropy", "communication_entropy_mean", "cyan"),
            ("Actor Loss", "actor_loss", "blue"),
            ("Critic Loss", "critic_loss", "red"),
            ("Perplexity", "communication_perplexity_mean", "magenta"),
            ("Codebook Usage", "communication_active_codebook_usage", "yellow")
        ]

        for label, key, color in tracked_metrics:
            last_entry = self.recent_history[-1] if self.recent_history else {}
            val = last_entry.get(key, "0")
            
            try:
                display_val = f"{float(val):.4f}"
            except (ValueError, TypeError):
                display_val = str(val)
                
            line = get_line_graph(self.recent_history, key)
            metrics_table.add_row(label, display_val, f"[{color}]{line}[/]")

        monitor_group = Group(
            f"[bold yellow]Status:[/] {status}",
            " ",
            progress,
            " ",
            metrics_table
        )
        
        monitor_panel = Panel(
            monitor_group, 
            title=f"[bold]Training Monitor - Step {timestep}[/]", 
            border_style="blue"
        )

        config_panel = self.generate_config_table()

        return Columns([config_panel, monitor_panel], expand=True)


    def close(self):
        self.live.stop()
        self.file.close()

class ObsLogger:

    def __init__(self, save_file):

        self.save_file = save_file

        file_exists = os.path.exists(self.save_file)

        if file_exists:
            os.remove(self.save_file)

        self.file = open(self.save_file, "a")
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

        self.save_files = [open(os.path.join(save_folder, f"comms_{i}.csv"), "a") for i in range(vocab_size)]

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