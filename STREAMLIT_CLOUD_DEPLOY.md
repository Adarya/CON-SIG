# Streamlit Cloud Deployment Guide

## 🚀 Deploy CONSIG to Streamlit Cloud

### Step 1: Prepare Repository
Ensure these files are in your repository root:
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System packages (apt-get)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `app.py` - Main application

### Step 2: Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Connect your GitHub account**
3. **Select repository**: `Adarya/CON-SIG`
4. **Set main file path**: `app.py`
5. **Advanced settings**:
   - Python version: `3.9`
   - Requirements file: `requirements.txt`
   - Packages file: `packages.txt`

### Step 3: Wait for Deployment
- Initial deployment takes 5-10 minutes
- Streamlit Cloud will install all dependencies
- You'll get a public URL when ready

## 🔧 Troubleshooting Streamlit Cloud

### Common Issues:

#### 1. **Dependency Errors**
- **Problem**: Missing Python packages
- **Solution**: Ensure `requirements.txt` has exact versions
- **Check**: Logs in Streamlit Cloud dashboard

#### 2. **System Package Errors**  
- **Problem**: Missing system libraries
- **Solution**: Add to `packages.txt`
- **Note**: Uses Ubuntu/apt-get packages

#### 3. **Memory Limits**
- **Problem**: Application crashes during bootstrap analysis
- **Solution**: Reduce bootstrap iterations in the app

#### 4. **File Size Limits**
- **Problem**: Large files fail to upload
- **Solution**: Streamlit Cloud has file size limits

### App Settings for Cloud:
```python
# In app.py, add cloud-specific settings:
if 'streamlit' in str(os.environ.get('PATH', '')):
    # Running on Streamlit Cloud
    st.set_page_config(
        page_title="CONSIG",
        layout="wide",
        initial_sidebar_state="expanded"
    )
```

## 📊 Performance Notes

### Streamlit Cloud Limitations:
- **Memory**: ~1GB RAM limit
- **CPU**: Shared resources
- **Storage**: Temporary (resets on restart)
- **Concurrent Users**: Limited

### Optimization Tips:
1. **Reduce bootstrap iterations** (50-100 instead of 200+)
2. **Cache results** using `@st.cache_data`
3. **Limit file sizes** (suggest <50MB)
4. **Use session state** to avoid recomputation

## 🔒 Security Notes

- **Public deployment**: Anyone with URL can access
- **Data handling**: Files are processed in cloud environment
- **Academic use**: Ensure compliance with data policies

## 📞 Support

If deployment fails:
1. **Check logs**: Streamlit Cloud dashboard shows detailed logs
2. **Verify files**: Ensure all required files are in repository
3. **Test locally**: Confirm app works with exact requirements
4. **Update dependencies**: Use tested version combinations

---

**Quick Deploy Button**: Once repository is ready, deployment is just a few clicks! 🚀