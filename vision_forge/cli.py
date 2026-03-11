"""Command Line Interface for Vision Forge."""

import asyncio
import click
import json
from pathlib import Path
from typing import Optional, List


@click.group()
@click.version_option(version="0.1.0", prog_name="vision-forge")
def cli():
    """
    Vision Forge - Intelligent Vision Reviewer

    Expert system for AI image evaluation and optimization.
    """
    pass


@cli.command()
@click.option('--prompt', '-p', required=True, help='User prompt for image generation/review')
@click.option('--image', '-i', multiple=True, help='Input image files (can specify multiple)')
@click.option('--output', '-o', default='output', help='Output directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def review(prompt, image, output, verbose):
    """Review/generate images based on prompt."""

    async def run_review():
        from vision_forge.core.orchestrator import WorkflowOrchestrator
        from vision_forge.experts.expert_registry import ExpertRegistry
        from vision_forge.memory.blackboard import SharedBlackboard
        from vision_forge.memory.manager import MemoryManager
        from vision_forge.services.router import ModelRouter

        # Initialize components
        registry = ExpertRegistry.from_directory("experts/static")
        blackboard = SharedBlackboard(persist_dir=Path(output) / "blackboard")
        memory = MemoryManager()
        router = ModelRouter.from_env()

        orchestrator = WorkflowOrchestrator(registry, blackboard, memory, router)

        # Load images if provided
        images = []
        for img_path in image:
            try:
                with open(img_path, 'rb') as f:
                    images.append(f.read())
            except FileNotFoundError:
                click.echo(f"Warning: Image not found: {img_path}", err=True)

        # Process request
        result = await orchestrator.process_request(
            user_prompt=prompt,
            images=images if images else None,
            context=None
        )

        # Output result
        if verbose:
            click.echo(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        else:
            click.echo(f"Status: {result['status']}")
            if result['status'] == 'success':
                click.echo(f"Task ID: {result['task_id']}")
                click.echo(f"Duration: {result.get('duration_sec', 0):.2f}s")
            elif result['status'] == 'rejected':
                click.echo(f"Violations: {result.get('violations', [])}")
            elif result['status'] == 'error':
                click.echo(f"Error: {result.get('error', 'Unknown error')}")

    asyncio.run(run_review())


@cli.command()
def experts():
    """List available experts."""
    from vision_forge.experts.expert_registry import ExpertRegistry

    registry = ExpertRegistry.from_directory("experts/static")

    click.echo(f"Loaded {len(registry.get_expert_ids())} static experts:\n")

    for expert_id in registry.get_expert_ids():
        config = registry._expert_configs.get(expert_id)
        if config:
            click.echo(f"  - {config.id}: {config.role}")
            click.echo(f"    Archetype: {config.archetype.value}")
            click.echo(f"    Load Strategy: {config.load_strategy.value}")
            click.echo()


@cli.command()
@click.option('--task-id', help='Task ID to check status')
def status(task_id):
    """Check task status."""
    if not task_id:
        click.echo("Please provide --task-id")
        return

    trace_path = Path("output/trace") / f"{task_id}.json"

    if trace_path.exists():
        with open(trace_path, 'r', encoding='utf-8') as f:
            trace = json.load(f)
        click.echo(f"Task found: {task_id}")
        click.echo(json.dumps(trace, indent=2, ensure_ascii=False))
    else:
        click.echo(f"Task not found: {task_id}")


@cli.command()
def config():
    """Show current configuration."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    click.echo("Current Configuration:\n")

    env_vars = [
        ('OPENAI_API_KEY', 'OpenAI API Key'),
        ('OPENAI_API_BASE', 'OpenAI API Base'),
        ('OPENAI_MODEL_NAME', 'OpenAI Model'),
        ('GOOGLE_API_KEY', 'Google API Key'),
        ('VERTEX_PROJECT_ID', 'Vertex AI Project ID'),
        ('VERTEX_LOCATION', 'Vertex AI Location'),
    ]

    for var, description in env_vars:
        value = os.getenv(var)
        if value:
            masked = value[:4] + '...' if len(value) > 4 else value
            click.echo(f"  {description}: {masked}")
        else:
            click.echo(f"  {description}: (not set)", fg='yellow')

    click.echo()


@cli.command()
@click.option('--provider', '-p', default='all', help='Provider to test (azure, vertex, all)')
@click.option('--prompt', '-p', 'test_prompt', default='Hello', help='Test prompt')
def test(provider, test_prompt):
    """Test LLM service connectivity."""

    async def run_test():
        from vision_forge.services.router import ModelRouter, TaskType
        from vision_forge.services.azure_openai import AzureOpenAIService
        from vision_forge.services.vertex_ai import VertexAIService

        router = ModelRouter.from_env()

        providers = []
        if provider in ['azure', 'all']:
            providers.append(('azure', 'Azure OpenAI'))
        if provider in ['vertex', 'all']:
            providers.append(('vertex', 'Vertex AI'))

        for provider_key, provider_name in providers:
            click.echo(f"\nTesting {provider_name}...")

            try:
                if provider_key == 'azure' and 'azure' in router.services:
                    service = router.services['azure']
                    response = await service.generate_text(test_prompt)
                    click.echo(f"  Status: OK")
                    click.echo(f"  Model: {response.model}")
                    click.echo(f"  Latency: {response.latency_ms}ms")
                elif provider_key == 'vertex' and 'vertex' in router.services:
                    service = router.services['vertex']
                    response = await service.generate_text(test_prompt)
                    click.echo(f"  Status: OK")
                    click.echo(f"  Model: {response.model}")
                    click.echo(f"  Latency: {response.latency_ms}ms")
                else:
                    click.echo(f"  Status: Not configured")
            except Exception as e:
                click.echo(f"  Error: {e}")

    asyncio.run(run_test())


@cli.command()
def stats():
    """Show system statistics."""
    from vision_forge.experts.expert_registry import ExpertRegistry
    from vision_forge.memory.blackboard import SharedBlackboard
    from vision_forge.memory.manager import MemoryManager

    registry = ExpertRegistry.from_directory("experts/static")
    blackboard = SharedBlackboard()
    memory = MemoryManager()

    click.echo("System Statistics:\n")
    click.echo("Experts:")
    stats = registry.get_stats()
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nMemory:")
    stats = memory.get_stats()
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nBlackboard:")
    stats = blackboard.get_stats()
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")


if __name__ == '__main__':
    cli()
