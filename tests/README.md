# Tests

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vision_forge --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests by keyword
pytest -k "expert" -v

# Run async tests
pytest -m asyncio -v
```

## Test Files

- `conftest.py` - Pytest fixtures and shared mocks
- `test_models.py` - Tests for Pydantic data models
- `test_config_loader.py` - Tests for YAML config loader
- `test_memory.py` - Tests for blackboard and memory system
- `test_experts.py` - Tests for expert classes and registry

## Fixtures

Available fixtures in `conftest.py`:

- `mock_blackboard` - Mock SharedBlackboard instance
- `mock_router` - Mock ModelRouter instance
- `sample_expert_config` - Sample ExpertConfig for testing
- `pm_expert_config` - Project Manager expert config
- `compliance_expert_config` - Compliance expert config
- `temp_output_dir` - Temporary output directory
- `sample_image_bytes` - Minimal PNG image bytes
