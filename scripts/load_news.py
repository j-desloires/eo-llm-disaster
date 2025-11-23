import sys
from typing import Any, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

from eo_disaster_analyzer.llm.schemas import DisasterEvent
from eo_disaster_analyzer.llm.clients import run_news_analysis_pipeline

app = typer.Typer()
console = Console()


def display_results(confirmed_events: List[tuple[DisasterEvent, dict[str, Any]]]):
    """Renders the list of confirmed disaster events to the console."""
    console.print(
        f"\n[bold green]✅ Found {len(confirmed_events)} confirmed disaster events:[/bold green]"
    )

    for i, (event, article) in enumerate(confirmed_events, 1):
        location_table = Table(title="Detected Locations", show_header=True, header_style="bold magenta")
        location_table.add_column("Name", style="cyan")
        location_table.add_column("Country", style="magenta")
        location_table.add_column("Lat/Lon", style="yellow")

        for loc in event.locations:
            coords = f"{loc.latitude:.4f}, {loc.longitude:.4f}" if loc.latitude and loc.longitude else "N/A"
            location_table.add_row(loc.name, loc.country or "N/A", coords)

        panel_content = f"[bold]Disaster Type:[/bold] {event.disaster_type}\n"
        panel_content += f"[bold]Summary:[/bold] {event.summary}\n"
        if event.event_date:
            panel_content += f"[bold]Date:[/bold] {event.event_date}\n"
        if event.casualties:
            panel_content += f"[bold]Casualties:[/bold] {event.casualties}\n"
        if event.source_url:
            panel_content += f"[bold]Source:[/bold] [link={event.source_url}]{event.source_url}[/link]"

        console.print(
            Panel(
                panel_content,
                title=f"[bold cyan]Event {i}: {event.title}[/bold cyan]",
                border_style="blue",
                expand=False,
            )
        )
        console.print(location_table)


@app.command()
def main(
        query: str = typer.Option(
            "flood OR flooding OR inundation",
            "--query",
            "-q",
            help="The search query for Google News.",
        ),
        period: str = typer.Option(
            "24h", "--period", "-p", help="Time period to search (e.g., '7d', '24h')."
        ),
        max_results: int = typer.Option(
            20, "--max-results", "-n", help="Max number of articles to fetch and process."
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose logging to see pipeline steps.",
        ),
):
    """
    Fetches and analyzes news articles for disaster events using an LLM pipeline.

    This script demonstrates the end-to-end process:
    1. Scrapes Google News for articles matching the query.
    2. Uses a cheap LLM call to pre-filter for relevant articles.
    3. Runs a detailed LLM analysis on the relevant articles to extract structured data.
    4. Prints the confirmed disaster events in a clean, readable format.
    """
    # Configure logging level based on the verbose flag
    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    console.rule("[bold green]📰 Starting News Analysis Pipeline[/bold green]")

    try:
        confirmed_pairs = run_news_analysis_pipeline(
            query=query, period=period, max_results=max_results
        )

        if not confirmed_pairs:
            console.print(
                "\n[bold yellow]Pipeline finished. No confirmed disaster events were found.[/bold yellow]"
            )
            return

        display_results(confirmed_pairs)

    except ValueError as e:
        # Catch missing API keys specifically
        if "OPENAI_API_KEY" in str(e):
            logger.error(f"Configuration Error: {e}")
            logger.error("Please make sure your OPENAI_API_KEY is set in your .env file.")
        else:
            logger.error(f"An unexpected value error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the script: {e}")


if __name__ == "__main__":
    app()