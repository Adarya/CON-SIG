# CONSIG Deployment Guide

## 🚀 Quick Deployment

### Local Development
```bash
git clone https://github.com/Adarya/CON-SIG.git
cd CON-SIG
./run_app.sh
```

The application will be available at `http://localhost:8501`

### Docker Deployment
```bash
git clone https://github.com/Adarya/CON-SIG.git
cd CON-SIG
docker-compose up --build
```

## ✅ Self-Contained Application

**No External Dependencies!** This application now includes:

- ✅ Complete CON_fitting framework
- ✅ CON_fitting_enhancements (bootstrap functionality)
- ✅ Consensus signatures reference data
- ✅ All required Python modules

## 🔧 Troubleshooting

### Import Issues
**Fixed!** The application is now self-contained and includes all required modules.

### Port Conflicts
If port 8501 is in use:
```bash
streamlit run app.py --server.port 8502
```

### Memory Issues
For large files or many bootstrap iterations:
- Reduce bootstrap iterations (50-200 is usually sufficient)
- Use Docker with increased memory limits

## 🌐 Cloud Deployment

### Streamlit Cloud
1. Fork the repository on GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from the repository

### Heroku
1. Create a Heroku app
2. Connect to your GitHub repository
3. Deploy using the included `Dockerfile`

### AWS/GCP/Azure
Use the provided `docker-compose.yml` for containerized deployment.

## 📊 Performance Notes

- **Bootstrap Analysis**: CPU-intensive, adjust iterations based on requirements
- **File Size**: Optimized for files up to 100MB
- **Concurrent Users**: Stateless design supports multiple users
- **Memory Usage**: Scales with sample count and bootstrap iterations

## 🔒 Security

For production deployment:
- Configure proper authentication if needed
- Use HTTPS for sensitive data
- Review firewall settings
- Monitor resource usage

## 📈 Monitoring

The application includes comprehensive logging. Monitor:
- Analysis completion times
- Error rates
- Memory usage
- File upload sizes

## 🆘 Support

If you encounter deployment issues:
1. Check the logs for specific error messages
2. Verify all dependencies are installed
3. Test with the provided example files
4. Open an issue on GitHub with detailed information