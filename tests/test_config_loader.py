"""Tests for config loader."""

import pytest
import yaml
from pathlib import Path

from vision_forge.core.config_loader import (
    load_expert_config,
    load_all_static_experts,
    save_expert_config
)
from vision_forge.core.models import Archetype, LoadStrategy


class TestLoadExpertConfig:
    """Test load_expert_config function."""

    def test_load_pm_config(self):
        config = load_expert_config("experts/static/pm.yml")
        assert config.id == "project_manager_01"
        assert config.role == "项目经理"
        assert config.archetype == Archetype.STATIC
        assert config.load_strategy == LoadStrategy.ALWAYS

    def test_load_compliance_config(self):
        config = load_expert_config("experts/static/compliance.yml")
        assert config.id == "compliance_legal_01"
        assert config.role == "合规与法律专家"
        assert config.archetype == Archetype.STATIC

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_expert_config("experts/static/nonexistent.yml")


class TestLoadAllStaticExperts:
    """Test load_all_static_experts function."""

    def test_load_all_static_experts(self):
        experts = load_all_static_experts("experts/static")
        # Should load all 10 static experts
        assert len(experts) == 10

    def test_load_all_static_experts_ids(self):
        experts = load_all_static_experts("experts/static")
        expert_ids = [e.id for e in experts]

        expected_ids = [
            "compliance_legal_01",
            "display_designer_01",
            "hr_expert_01",
            "image_editor_01",
            "interior_designer_01",
            "knowledge_admin_01",
            "photographer_01",
            "project_manager_01",
            "prompt_engineer_01",
            "visual_expert_01",
        ]

        for expected_id in expected_ids:
            assert expected_id in expert_ids

    def test_load_from_nonexistent_directory(self):
        experts = load_all_static_experts("nonexistent/directory")
        assert experts == []


class TestSaveExpertConfig:
    """Test save_expert_config function."""

    def test_save_and_reload(self, sample_expert_config, tmp_path):
        output_path = tmp_path / "test_expert.yml"
        save_expert_config(sample_expert_config, str(output_path))

        assert output_path.exists()

        # Load back
        loaded = load_expert_config(str(output_path))
        assert loaded.id == sample_expert_config.id
        assert loaded.role == sample_expert_config.role

    def test_save_creates_directory(self, sample_expert_config, tmp_path):
        output_path = tmp_path / "subdir" / "expert.yml"
        save_expert_config(sample_expert_config, str(output_path))

        assert output_path.exists()


class TestExpertYamlStructure:
    """Test YAML structure of expert configs."""

    def test_pm_yaml_structure(self):
        with open("experts/static/pm.yml", 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        assert 'expert' in data
        expert = data['expert']
        assert 'id' in expert
        assert 'role' in expert
        assert 'persona' in expert
        assert 'capabilities' in expert

    def test_compliance_yaml_structure(self):
        with open("experts/static/compliance.yml", 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        assert 'expert' in data
        expert = data['expert']
        assert 'veto_power' in expert
        assert expert['veto_power']['enabled'] is True
