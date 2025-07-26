# Contributing to CONSIG

We welcome contributions to CONSIG! This document provides guidelines for contributing to the project.

## 📋 Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of Streamlit and pandas
- Familiarity with CNA signature analysis concepts

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git fork https://github.com/Adarya/CON-SIG
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/[your-username]/CON-SIG
   cd CON-SIG
   ```

3. **Set up development environment**
   ```bash
   ./run_app.sh  # This will create venv and install dependencies
   ```

4. **Run tests**
   ```bash
   python test_complete_workflow.py
   ```

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing
- Add tests for new functionality
- Ensure all existing tests pass
- Test with both example data and real datasets
- Include edge case testing

### Documentation
- Update README.md if adding new features
- Add inline comments for complex logic
- Update INSTALLATION.md for setup changes

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - Operating system
   - Streamlit version

2. **Steps to reproduce**:
   - Exact steps taken
   - Input files used (if possible to share)
   - Error messages

3. **Expected vs actual behavior**

## ✨ Feature Requests

For feature requests, please:

1. Check if the feature already exists
2. Explain the use case and benefit
3. Provide examples if possible
4. Consider implementation complexity

## 📝 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   python test_complete_workflow.py
   python test_backend.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

## 🏗️ Development Areas

We welcome contributions in these areas:

### Core Functionality
- Additional deconvolution methods
- Performance optimizations
- Error handling improvements

### User Interface
- UI/UX improvements
- Additional visualization options
- Better error messages

### Documentation
- Tutorial improvements
- Example datasets
- API documentation

### Testing
- Additional test cases
- Integration tests
- Performance testing

## 🔒 Security

If you discover security vulnerabilities, please:

1. **Do not** open a public issue
2. Email the maintainers privately
3. Provide detailed description of the vulnerability
4. Allow time for the issue to be addressed

## 📄 License

By contributing, you agree that your contributions will be licensed under the same academic license as the project.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Academic publications (where appropriate)

## 📞 Questions?

If you have questions about contributing:

- Open a GitHub issue with the "question" label
- Check existing issues and discussions
- Review the documentation

Thank you for contributing to CONSIG! 🧬