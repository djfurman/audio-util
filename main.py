import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Annotated, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from audio import AudioProcessor


class FolderSelector:
    """GUI folder selection utility."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

    def select_input_folder(self) -> Optional[str]:
        """Open a folder selection dialog for input directory."""
        folder = filedialog.askdirectory(
            title="Select Input Directory (containing WAV files)", mustexist=True
        )
        return folder if folder else None

    def select_output_folder(self) -> Optional[str]:
        """Open a folder selection dialog for output directory."""
        folder = filedialog.askdirectory(
            title="Select Output Directory (for processed files)"
        )
        return folder if folder else None

    def show_completion_message(self, output_dir: str):
        """Show completion message with output directory."""
        messagebox.showinfo(
            "Processing Complete",
            f"Processing completed successfully!\n\nOutput files saved to:\n{output_dir}",
        )

    def show_error_message(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)

    def select_noise_files(self) -> List[str]:
        """Open a file selection dialog for multiple noise files."""
        file_paths = filedialog.askopenfilenames(
            title="Select Noise Files (.wav)",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        return list(file_paths) if file_paths else []


# Initialize Typer app
app = typer.Typer(
    help="WAV File Consolidator and Filter Tool for Scientific Experiments",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def process(
    input_dir: Annotated[
        Optional[str], typer.Argument(help="Input directory containing WAV files")
    ] = None,
    output_dir: Annotated[
        Optional[str], typer.Argument(help="Output directory for processed files")
    ] = None,
    preset: Annotated[str, typer.Option(help="Filter preset to apply")] = "scientific",
    gap: Annotated[float, typer.Option(help="Gap between files in seconds")] = 0.5,
    gui: Annotated[
        bool, typer.Option("--gui", help="Use GUI folder selection")
    ] = False,
    noise_profile: Annotated[
        Optional[str], typer.Option(help="Path to noise profile file")
    ] = None,
):
    """
    Process WAV files: consolidate and apply filters.

    Available presets: scientific, noise_reduction, vocal_enhancement, adaptive_noise_reduction
    """

    # Display welcome message
    console.print(
        Panel.fit(
            "[bold blue]🎵 WAV File Consolidator & Filter Tool[/bold blue]\n"
            "[dim]For Scientific Experiments[/dim]",
            border_style="blue",
        )
    )

    processor = AudioProcessor()

    # Handle GUI mode or missing arguments
    if gui or not input_dir or not output_dir:
        folder_selector = FolderSelector()

        if not input_dir:
            console.print(
                "\n[yellow]Please select the input directory containing WAV files...[/yellow]"
            )
            input_dir = folder_selector.select_input_folder()
            if not input_dir:
                console.print("[red]No input directory selected. Exiting.[/red]")
                return

        if not output_dir:
            console.print(
                "\n[yellow]Please select the output directory for processed files...[/yellow]"
            )
            output_dir = folder_selector.select_output_folder()
            if not output_dir:
                console.print("[red]No output directory selected. Exiting.[/red]")
                return

        # Handle noise profile selection
        if gui and not noise_profile:
            if Confirm.ask(
                "\n[yellow]Do you want to use a noise profile for better noise reduction?[/yellow]",
                default=False,
            ):
                console.print(
                    "[yellow]Please select the noise profile file...[/yellow]"
                )
                noise_profile = folder_selector.select_noise_profile()
                if noise_profile:
                    console.print(
                        f"[green]Selected noise profile: {os.path.basename(noise_profile)}[/green]"
                    )
                else:
                    console.print(
                        "[yellow]No noise profile selected. Proceeding without noise profiling.[/yellow]"
                    )

    # Validate preset
    valid_presets = [
        "scientific",
        "noise_reduction",
        "vocal_enhancement",
        "adaptive_noise_reduction",
    ]
    if preset not in valid_presets:
        console.print(
            f"[red]Invalid preset '{preset}'. Valid options: {', '.join(valid_presets)}[/red]"
        )
        preset = Prompt.ask(
            "Please select a preset", choices=valid_presets, default="scientific"
        )

    # Display processing parameters
    params_table = Table(title="Processing Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Input Directory", input_dir)
    params_table.add_row("Output Directory", output_dir)
    params_table.add_row("Filter Preset", preset)
    params_table.add_row("Gap Between Files", f"{gap} seconds")
    params_table.add_row("Noise Profile", noise_profile if noise_profile else "None")

    console.print(params_table)

    # Confirm processing
    if not Confirm.ask("\n[bold]Proceed with processing?[/bold]", default=True):
        console.print("[yellow]Processing cancelled.[/yellow]")
        return

    # Process the directory
    success = processor.process_directory(
        input_dir, output_dir, preset, gap, noise_profile
    )

    if success:
        console.print("\n[green]✅ Processing completed successfully![/green]")
        console.print(f"[blue]📁 Output files saved to: {output_dir}[/blue]")

        # Show GUI completion message if GUI was used
        if gui:
            folder_selector = FolderSelector()
            folder_selector.show_completion_message(output_dir)
    else:
        console.print("\n[red]❌ Processing failed. Check the logs for details.[/red]")
        if gui:
            folder_selector = FolderSelector()
            folder_selector.show_error_message(
                "Processing failed. Check the console for details."
            )


@app.command()
def interactive():
    """
    Run in interactive mode with prompts for all parameters.
    """
    console.print(
        Panel.fit(
            "[bold green]🎛️ Interactive Mode[/bold green]\n"
            "[dim]Step-by-step configuration[/dim]",
            border_style="green",
        )
    )

    # Get input directory
    use_gui = Confirm.ask("Use GUI for folder selection?", default=True)

    if use_gui:
        folder_selector = FolderSelector()
        console.print("\n[yellow]Select input directory...[/yellow]")
        input_dir = folder_selector.select_input_folder()
        if not input_dir:
            console.print("[red]No input directory selected. Exiting.[/red]")
            return

        console.print("\n[yellow]Select output directory...[/yellow]")
        output_dir = folder_selector.select_output_folder()
        if not output_dir:
            console.print("[red]No output directory selected. Exiting.[/red]")
            return
    else:
        input_dir = Prompt.ask("Enter input directory path")
        output_dir = Prompt.ask("Enter output directory path")

    # Get other parameters
    preset = Prompt.ask(
        "Select filter preset",
        choices=["scientific", "noise_reduction", "vocal_enhancement"],
        default="scientific",
    )

    gap = float(Prompt.ask("Gap between files (seconds)", default="0.5"))

    # Process
    processor = AudioProcessor()
    success = processor.process_directory(input_dir, output_dir, preset, gap)

    if success and use_gui:
        folder_selector.show_completion_message(output_dir)


if __name__ == "__main__":
    app()
