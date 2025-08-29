# Contributing to DINOv3-Based Alpha Matting

We welcome contributions from the community! This document provides guidelines for contributing to this project.

## 🚀 Ways to Contribute

### Bug Reports and Feature Requests
- **Bug Reports**: Use [GitHub Issues](https://github.com/nikhiliit/Dinov3_matting/issues) to report bugs
- **Feature Requests**: Suggest new features or improvements
- **Questions**: Ask questions about usage or implementation

### Code Contributions
- **Pull Requests**: Submit PRs for bug fixes, features, or documentation
- **Documentation**: Improve README, add examples, or fix typos
- **Testing**: Add test cases or improve existing tests

## 📋 Development Setup

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 2.0.0
CUDA >= 11.0 (recommended for GPU training)
```

### Local Development
```bash
# Clone the repository
git clone https://github.com/nikhiliit/Dinov3_matting.git
cd Dinov3_matting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing Your Changes
```bash
# Run demo (no dataset required)
python demo.py

# Run with sample data
python train.py --config configs/dinov3_alpha_config.yaml
```

## 🎯 Code Style Guidelines

### Python Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep line length under 88 characters

### Example Code Style
```python
def process_alpha_matte(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Process predicted alpha matte with thresholding.

    Args:
        pred_alpha: Predicted alpha values [B, 1, H, W]
        target_alpha: Ground truth alpha values [B, 1, H, W]
        threshold: Threshold for binary masking

    Returns:
        Processed alpha matte with applied threshold
    """
    return torch.where(pred_alpha > threshold, pred_alpha, torch.zeros_like(pred_alpha))
```

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in imperative mood
- Keep first line under 50 characters
- Add detailed description if needed

**Examples:**
```
feat: add boundary refinement module
fix: resolve memory leak in data loader
docs: update installation instructions
refactor: simplify loss computation logic
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src --cov-report=html
```

### Writing Tests
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Include docstrings for complex tests

### Test Coverage
- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests for key workflows

## 📖 Documentation

### Code Documentation
- All public functions must have docstrings
- Include parameter types and descriptions
- Document return values and exceptions
- Use Google/NumPy docstring format

### README Updates
- Keep README.md up to date with new features
- Update installation instructions as needed
- Add examples for new functionality
- Update citation information

## 🔧 Pull Request Process

### Before Submitting
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Make** your changes following the guidelines above
4. **Test** your changes thoroughly
5. **Update** documentation if needed
6. **Commit** with clear messages

### PR Checklist
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes without discussion
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

### PR Template
```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Code refactor

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings or errors
- [ ] All tests pass
```

## 🎓 Research Contributions

### Novel Contributions
- **New Architectures**: Propose novel model architectures
- **Loss Functions**: Develop improved loss functions
- **Training Strategies**: Experiment with new training approaches
- **Evaluation Metrics**: Propose better evaluation methods

### Reproducing Results
- Provide clear instructions for reproducing results
- Include hyperparameters and random seeds
- Document dataset preprocessing steps
- Share trained model checkpoints

### Paper Writing
- Follow academic writing standards
- Include proper citations and references
- Provide comprehensive experimental analysis
- Compare with state-of-the-art methods

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions
- **Email**: For private communications

### Support Guidelines
- Check existing issues before creating new ones
- Provide minimal reproducible examples for bugs
- Include system information and error messages
- Be respectful and constructive in communications

## 🙏 Recognition

Contributors will be:
- Listed in the repository's CONTRIBUTORS.md file
- Acknowledged in research publications
- Invited to join the project maintainer team (for significant contributions)

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to DINOv3-Based Alpha Matting! 🎨✨
