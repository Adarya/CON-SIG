# CONSIG Installation Guide

## Quick Start

The easiest way to get CONSIG running is using the provided launch script:

```bash
cd CON_fitting_app
./run_app.sh
```

This will automatically:
- Create a virtual environment
- Install all dependencies
- Launch the application

## Manual Installation

If you prefer to install manually:

### 1. Prerequisites

- Python 3.9 or higher
- pip package manager

### 2. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_app.txt
```

### 3. Launch Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Docker Installation

For containerized deployment:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or manually with Docker
docker build -t consig-app .
docker run -p 8501:8501 consig-app
```

## Troubleshooting

### Common Issues

1. **Python Version**: Ensure you have Python 3.9+
   ```bash
   python --version
   ```

2. **Port Already in Use**: Change the port
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **Permission Denied**: Make the run script executable
   ```bash
   chmod +x run_app.sh
   ```

4. **Import Errors**: Ensure you're in the correct directory
   ```bash
   cd CON_fitting_app
   python -c "import backend; print('Backend imported successfully')"
   ```

### Verification

Test that everything is working:

```bash
# Test backend functionality
python test_backend.py

# Test basic imports
python -c "from backend import get_example_data; print('Success!')"
```

## What's Included

After installation, you'll have:

- ✅ Web application (`app.py`)
- ✅ Backend processing (`backend.py`)
- ✅ Visualization tools (`plotting.py`)
- ✅ Example data files (`examples/`)
- ✅ Docker configuration
- ✅ Documentation

## Next Steps

1. **Start the application**: `./run_app.sh`
2. **Upload test data**: Try `examples/example_matrix.tsv`
3. **Run analysis**: Configure parameters and click "Run Analysis"
4. **View results**: Explore the results tabs

## Support

If you encounter issues:
- Check the terminal output for error messages
- Verify all dependencies are installed
- Try the example files first
- Ensure CON_fitting modules are accessible

For additional help, check the main README.md file.