# Folder Size Explorer

A Streamlit web application for analyzing Azure Blob Storage inventory files to visualize folder sizes hierarchically.

## Features

- 📁 Upload Parquet or CSV inventory files
- 📊 Interactive treemap and sunburst visualizations
- 🔍 Advanced filtering (depth, size, search)
- 📋 Sortable data table with AgGrid
- 💾 Export results as CSV or JSON
- 🚀 Optimized for large files (Parquet recommended)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## File Requirements

Your inventory file should contain:
- **Name** (required): Full blob path (e.g., `folder/subfolder/file.txt`)
- **Content-Length** (required): File size in bytes
- **LastModified** (optional): Modification timestamp
- **BlobType** (optional): Type of blob
- **Container** (optional): Container name

## Performance Tips

- Use **Parquet** format for best performance with large files
- The app automatically selects only needed columns from Parquet files
- CSV files work but may be slower for very large inventories

## Usage

1. Upload your Blob Inventory file using the sidebar
2. Adjust filters (folder depth, minimum size, search)
3. Explore the treemap and sunburst visualizations
4. Review the detailed table view
5. Download aggregated results as needed