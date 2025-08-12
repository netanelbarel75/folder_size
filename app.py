import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import json
import io
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
import time
import gc
import psutil
import os
from collections import defaultdict

# Configure Streamlit page
st.set_page_config(
    page_title="Folder Size Explorer",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to increase upload limit programmatically (may not work in all environments)
try:
    import streamlit.web.server.server
    streamlit.web.server.server.UPLOAD_FILE_SIZE_LIMIT_MB = 2048
except:
    pass

# Constants for large file handling
CHUNK_SIZE = 50000  # Process 50K rows at a time for CSV
PARQUET_BATCH_SIZE = 100000  # Read 100K rows at a time for Parquet
MAX_MEMORY_PERCENT = 80  # Alert if memory usage exceeds this
PROGRESS_UPDATE_INTERVAL = 10000  # Update progress every N rows

# Utility Functions
def get_memory_usage():
    """Get current memory usage percentage."""
    return psutil.Process(os.getpid()).memory_percent()

def bytes_to_readable(bytes_val: float) -> str:
    """Convert bytes to human-readable format with 2 decimals."""
    if bytes_val < 1024:
        return f"{bytes_val:.2f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.2f} MB"
    elif bytes_val < 1024**4:
        return f"{bytes_val/(1024**3):.2f} GB"
    else:
        return f"{bytes_val/(1024**4):.2f} TB"

def split_path_to_prefixes(path: str, max_depth: int = 10) -> List[str]:
    """Split a path into all prefix levels up to max_depth."""
    if pd.isna(path) or not path:
        return ["(root)"]
    
    # Handle paths that don't contain '/'
    if '/' not in path:
        return ["(root)"]
    
    parts = path.split('/')
    # Remove empty parts and file name (last part)
    parts = [p for p in parts[:-1] if p]  # Remove last part (filename) and empty strings
    
    if not parts:
        return ["(root)"]
    
    prefixes = []
    for i in range(min(len(parts), max_depth)):
        prefix = '/'.join(parts[:i+1])
        prefixes.append(prefix)
    
    return prefixes

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and validate required columns."""
    # Normalize column names to handle case differences
    column_mapping = {}
    for col in df.columns:
        lower_col = col.lower()
        if lower_col in ['name']:
            column_mapping[col] = 'Name'
        elif lower_col in ['contentlength', 'content_length', 'content-length', 'size']:
            column_mapping[col] = 'Content-Length'
        elif lower_col in ['lastmodified', 'last_modified', 'last-modified']:
            column_mapping[col] = 'LastModified'
        elif lower_col in ['blobtype', 'blob_type', 'blob-type']:
            column_mapping[col] = 'BlobType'
        elif lower_col in ['container']:
            column_mapping[col] = 'Container'
    
    df = df.rename(columns=column_mapping)
    
    # Validate required columns
    required_cols = ['Name', 'Content-Length']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def get_file_info(uploaded_file) -> Dict:
    """Get file information including estimated row count."""
    file_size = len(uploaded_file.getvalue())
    file_type = 'parquet' if uploaded_file.name.endswith('.parquet') else 'csv'
    
    # Estimate row count based on file size (rough estimates)
    if file_type == 'parquet':
        estimated_rows = file_size // 100  # Assume ~100 bytes per row (compressed)
    else:
        estimated_rows = file_size // 200  # Assume ~200 bytes per row for CSV
    
    return {
        'size_bytes': file_size,
        'size_readable': bytes_to_readable(file_size),
        'type': file_type,
        'estimated_rows': estimated_rows
    }

def get_file_info_local(filename: str) -> Dict:
    """Get file information for local files."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    
    file_size = os.path.getsize(filename)
    file_type = 'parquet' if filename.endswith('.parquet') else 'csv'
    
    # Estimate row count based on file size (rough estimates)
    if file_type == 'parquet':
        estimated_rows = file_size // 100  # Assume ~100 bytes per row (compressed)
    else:
        estimated_rows = file_size // 200  # Assume ~200 bytes per row for CSV
    
    return {
        'size_bytes': file_size,
        'size_readable': bytes_to_readable(file_size),
        'type': file_type,
        'estimated_rows': estimated_rows
    }

def process_local_parquet_in_batches(filename: str, max_depth: int, include_root: bool, 
                                    progress_placeholder) -> Dict[str, Dict]:
    """Process large local Parquet file in batches."""
    aggregated_data = defaultdict(lambda: {'bytes': 0, 'file_count': 0, 'level': 0})
    
    try:
        parquet_file = pq.ParquetFile(filename)
        total_rows = parquet_file.metadata.num_rows
        
        # Get available columns and select only needed ones
        schema = parquet_file.schema
        available_cols = [field.name for field in schema]
        needed_cols = []
        for col in available_cols:
            lower_col = col.lower()
            if lower_col in ['name', 'contentlength', 'content_length', 'content-length', 'size']:
                needed_cols.append(col)
        
        processed_rows = 0
        batch_num = 0
        
        # Process in batches
        for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE, 
                                             columns=needed_cols if needed_cols else None):
            batch_df = batch.to_pandas()
            batch_df = normalize_columns(batch_df)
            
            # Process this batch
            for _, row in batch_df.iterrows():
                path = row['Name']
                size = row['Content-Length']
                
                if pd.isna(size) or size < 0:
                    continue
                    
                prefixes = split_path_to_prefixes(path, max_depth)
                
                for level, prefix in enumerate(prefixes, 1):
                    if not include_root and prefix == "(root)":
                        continue
                    
                    aggregated_data[prefix]['bytes'] += size
                    aggregated_data[prefix]['file_count'] += 1
                    aggregated_data[prefix]['level'] = level
            
            processed_rows += len(batch_df)
            batch_num += 1
            
            # Update progress
            progress = min(processed_rows / total_rows, 1.0)
            memory_usage = get_memory_usage()
            
            progress_placeholder.progress(
                progress, 
                text=f"Processing batch {batch_num} | {processed_rows:,}/{total_rows:,} rows ({progress:.1%}) | Memory: {memory_usage:.1f}%"
            )
            
            # Memory check
            if memory_usage > MAX_MEMORY_PERCENT:
                st.warning(f"⚠️ High memory usage ({memory_usage:.1f}%). Consider using a smaller file or increasing system RAM.")
            
            # Force garbage collection periodically
            if batch_num % 10 == 0:
                gc.collect()
    
    except Exception as e:
        raise ValueError(f"Error processing local Parquet file: {str(e)}")
    
    return dict(aggregated_data)

def process_local_csv_in_chunks(filename: str, max_depth: int, include_root: bool,
                               progress_placeholder) -> Dict[str, Dict]:
    """Process large local CSV file in chunks."""
    aggregated_data = defaultdict(lambda: {'bytes': 0, 'file_count': 0, 'level': 0})
    
    try:
        # First pass: count total rows for progress tracking
        with open(filename, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract header
        
        processed_rows = 0
        chunk_num = 0
        
        # Process in chunks
        chunk_iter = pd.read_csv(filename, chunksize=CHUNK_SIZE)
        
        for chunk_df in chunk_iter:
            chunk_df = normalize_columns(chunk_df)
            chunk_num += 1
            
            # Process this chunk
            for _, row in chunk_df.iterrows():
                path = row['Name']
                size = row['Content-Length']
                
                if pd.isna(size) or size < 0:
                    continue
                    
                prefixes = split_path_to_prefixes(path, max_depth)
                
                for level, prefix in enumerate(prefixes, 1):
                    if not include_root and prefix == "(root)":
                        continue
                    
                    aggregated_data[prefix]['bytes'] += size
                    aggregated_data[prefix]['file_count'] += 1
                    aggregated_data[prefix]['level'] = level
            
            processed_rows += len(chunk_df)
            
            # Update progress
            progress = min(processed_rows / total_rows, 1.0) if total_rows > 0 else 0
            memory_usage = get_memory_usage()
            
            progress_placeholder.progress(
                progress,
                text=f"Processing chunk {chunk_num} | {processed_rows:,}/{total_rows:,} rows ({progress:.1%}) | Memory: {memory_usage:.1f}%"
            )
            
            # Memory check
            if memory_usage > MAX_MEMORY_PERCENT:
                st.warning(f"⚠️ High memory usage ({memory_usage:.1f}%). Consider using a smaller file or increasing system RAM.")
            
            # Force garbage collection periodically
            if chunk_num % 5 == 0:
                gc.collect()
    
    except Exception as e:
        raise ValueError(f"Error processing local CSV file: {str(e)}")
    
    return dict(aggregated_data)

def process_parquet_in_batches(uploaded_file, max_depth: int, include_root: bool, 
                              progress_placeholder) -> Dict[str, Dict]:
    """Process large Parquet file in batches."""
    aggregated_data = defaultdict(lambda: {'bytes': 0, 'file_count': 0, 'level': 0})
    
    try:
        parquet_file = pq.ParquetFile(uploaded_file)
        total_rows = parquet_file.metadata.num_rows
        
        # Get available columns and select only needed ones
        schema = parquet_file.schema
        available_cols = [field.name for field in schema]
        needed_cols = []
        for col in available_cols:
            lower_col = col.lower()
            if lower_col in ['name', 'contentlength', 'content_length', 'content-length', 'size']:
                needed_cols.append(col)
        
        processed_rows = 0
        batch_num = 0
        
        # Process in batches
        for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE, 
                                             columns=needed_cols if needed_cols else None):
            batch_df = batch.to_pandas()
            batch_df = normalize_columns(batch_df)
            
            # Process this batch
            for _, row in batch_df.iterrows():
                path = row['Name']
                size = row['Content-Length']
                
                if pd.isna(size) or size < 0:
                    continue
                    
                prefixes = split_path_to_prefixes(path, max_depth)
                
                for level, prefix in enumerate(prefixes, 1):
                    if not include_root and prefix == "(root)":
                        continue
                    
                    aggregated_data[prefix]['bytes'] += size
                    aggregated_data[prefix]['file_count'] += 1
                    aggregated_data[prefix]['level'] = level
            
            processed_rows += len(batch_df)
            batch_num += 1
            
            # Update progress
            progress = min(processed_rows / total_rows, 1.0)
            memory_usage = get_memory_usage()
            
            progress_placeholder.progress(
                progress, 
                text=f"Processing batch {batch_num} | {processed_rows:,}/{total_rows:,} rows ({progress:.1%}) | Memory: {memory_usage:.1f}%"
            )
            
            # Memory check
            if memory_usage > MAX_MEMORY_PERCENT:
                st.warning(f"⚠️ High memory usage ({memory_usage:.1f}%). Consider using a smaller file or increasing system RAM.")
            
            # Force garbage collection periodically
            if batch_num % 10 == 0:
                gc.collect()
    
    except Exception as e:
        raise ValueError(f"Error processing Parquet file: {str(e)}")
    
    return dict(aggregated_data)

def process_csv_in_chunks(uploaded_file, max_depth: int, include_root: bool,
                         progress_placeholder) -> Dict[str, Dict]:
    """Process large CSV file in chunks."""
    aggregated_data = defaultdict(lambda: {'bytes': 0, 'file_count': 0, 'level': 0})
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    try:
        # First pass: count total rows for progress tracking
        total_rows = sum(1 for _ in uploaded_file) - 1  # Subtract header
        uploaded_file.seek(0)
        
        processed_rows = 0
        chunk_num = 0
        
        # Process in chunks
        chunk_iter = pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE)
        
        for chunk_df in chunk_iter:
            chunk_df = normalize_columns(chunk_df)
            chunk_num += 1
            
            # Process this chunk
            for _, row in chunk_df.iterrows():
                path = row['Name']
                size = row['Content-Length']
                
                if pd.isna(size) or size < 0:
                    continue
                    
                prefixes = split_path_to_prefixes(path, max_depth)
                
                for level, prefix in enumerate(prefixes, 1):
                    if not include_root and prefix == "(root)":
                        continue
                    
                    aggregated_data[prefix]['bytes'] += size
                    aggregated_data[prefix]['file_count'] += 1
                    aggregated_data[prefix]['level'] = level
            
            processed_rows += len(chunk_df)
            
            # Update progress
            progress = min(processed_rows / total_rows, 1.0) if total_rows > 0 else 0
            memory_usage = get_memory_usage()
            
            progress_placeholder.progress(
                progress,
                text=f"Processing chunk {chunk_num} | {processed_rows:,}/{total_rows:,} rows ({progress:.1%}) | Memory: {memory_usage:.1f}%"
            )
            
            # Memory check
            if memory_usage > MAX_MEMORY_PERCENT:
                st.warning(f"⚠️ High memory usage ({memory_usage:.1f}%). Consider using a smaller file or increasing system RAM.")
            
            # Force garbage collection periodically
            if chunk_num % 5 == 0:
                gc.collect()
    
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")
    
    return dict(aggregated_data)

def convert_aggregated_to_dataframe(aggregated_data: Dict[str, Dict]) -> pd.DataFrame:
    """Convert aggregated dictionary to DataFrame."""
    if not aggregated_data:
        return pd.DataFrame(columns=['level', 'prefix', 'bytes', 'mb', 'gb', 'file_count'])
    
    rows = []
    for prefix, data in aggregated_data.items():
        rows.append({
            'level': data['level'],
            'prefix': prefix,
            'bytes': data['bytes'],
            'file_count': data['file_count']
        })
    
    df = pd.DataFrame(rows)
    df['mb'] = df['bytes'] / (1024**2)
    df['gb'] = df['bytes'] / (1024**3)
    
    return df.sort_values(['level', 'bytes'], ascending=[True, False])

def create_treemap(df: pd.DataFrame, max_items: int = 30) -> go.Figure:
    """Create cleaner treemap visualization."""
    if df.empty:
        return go.Figure().add_annotation(text="No data to display", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Take top items by size - focus on level 1 and 2 folders for cleaner display
    plot_df = df.nlargest(max_items, 'bytes').copy()
    
    # Create simple single-level treemap for better readability
    # Group similar folders and show clean hierarchy
    treemap_data = []
    
    for _, row in plot_df.iterrows():
        prefix = row['prefix']
        size = row['bytes']
        
        if prefix == "(root)":
            treemap_data.append({
                'ids': prefix,
                'labels': f"(root)<br><b>{bytes_to_readable(size)}</b>",
                'parents': "",
                'values': size
            })
        else:
            parts = prefix.split('/')
            # Use only the first part as main category for cleaner view
            main_folder = parts[0]
            
            # Create readable label
            if len(parts) == 1:
                label = f"{main_folder}<br><b>{bytes_to_readable(size)}</b>"
            else:
                # Show path with ellipsis if too long
                display_path = prefix if len(prefix) <= 25 else f"{main_folder}/.../{parts[-1]}"
                label = f"{display_path}<br><b>{bytes_to_readable(size)}</b>"
            
            treemap_data.append({
                'ids': prefix,
                'labels': label,
                'parents': "",
                'values': size
            })
    
    # Convert to DataFrame for plotly
    treemap_df = pd.DataFrame(treemap_data)
    
    # Create treemap using graph_objects for better control
    fig = go.Figure(go.Treemap(
        ids=treemap_df['ids'],
        labels=treemap_df['labels'],
        parents=treemap_df['parents'],
        values=treemap_df['values'],
        textinfo="label",
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Size: %{value:,.0f} bytes<br>Percentage: %{percentParent}<extra></extra>',
        maxdepth=2,
        pathbar={"visible": False}  # Hide the path bar for cleaner look
    ))
    
    fig.update_layout(
        title=f"Top {len(plot_df)} Folders by Size",
        height=500,
        font=dict(size=10)
    )
    
    return fig

def create_hierarchy_tree(df: pd.DataFrame) -> str:
    """Create a file system hierarchy tree view."""
    if df.empty:
        return "No data to display"
    
    # Build tree structure
    tree_data = {}
    
    # Sort by size (largest first) and limit for readability
    sorted_df = df.nlargest(100, 'bytes') if len(df) > 100 else df
    
    for _, row in sorted_df.iterrows():
        prefix = row['prefix']
        size = row['bytes']
        file_count = row['file_count']
        
        if prefix == "(root)":
            tree_data["(root)"] = {
                'size': size,
                'files': file_count,
                'children': {}
            }
        else:
            parts = prefix.split('/')
            current = tree_data
            
            # Build nested structure
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {
                        'size': 0,
                        'files': 0,
                        'children': {}
                    }
                
                # Update size and file count for this level
                if i == len(parts) - 1:  # Last part
                    current[part]['size'] = size
                    current[part]['files'] = file_count
                
                current = current[part]['children']
    
    # Convert to HTML tree
    def build_html_tree(data, level=0, max_level=4):
        if level > max_level:
            return ""
        
        html = ""
        indent = "&nbsp;" * (level * 4)
        
        # Sort by size
        sorted_items = sorted(data.items(), key=lambda x: x[1]['size'], reverse=True)
        
        for name, info in sorted_items[:20]:  # Limit to top 20 at each level
            size_str = bytes_to_readable(info['size'])
            files_str = f"{info['files']:,}" if info['files'] > 0 else ""
            
            # Icon based on level
            if level == 0:
                icon = "📁"
            elif level == 1:
                icon = "📂"
            else:
                icon = "📄"
            
            html += f"{indent}{icon} <b>{name}</b> "
            if info['size'] > 0:
                html += f"<span style='color: #0066cc;'>({size_str}"
                if files_str:
                    html += f", {files_str} files"
                html += ")</span>"
            html += "<br>"
            
            # Add children if any
            if info['children'] and level < max_level:
                html += build_html_tree(info['children'], level + 1, max_level)
        
        return html
    
    return build_html_tree(tree_data)

def create_folder_table(df: pd.DataFrame, max_depth_filter: int) -> pd.DataFrame:
    """Create a clean folder table grouped by depth level."""
    if df.empty:
        return pd.DataFrame()
    
    # Filter by max depth
    filtered_df = df[df['level'] <= max_depth_filter].copy()
    
    # Group by level for better organization
    tables_by_level = {}
    
    for level in sorted(filtered_df['level'].unique()):
        level_df = filtered_df[filtered_df['level'] == level].nlargest(20, 'bytes')
        level_df['Size'] = level_df['bytes'].apply(bytes_to_readable)
        level_df['Files'] = level_df['file_count'].apply(lambda x: f"{x:,}")
        
        # Clean up prefix display for each level
        level_df['Folder'] = level_df['prefix'].apply(lambda x: 
            x.split('/')[-1] if x != "(root)" and '/' in x else x
        )
        
        display_cols = ['Folder', 'Size', 'Files']
        tables_by_level[f"Level {level}"] = level_df[display_cols]
    
    return tables_by_level

def create_sunburst(df: pd.DataFrame, max_items: int = 200) -> go.Figure:
    """Create sunburst chart for hierarchical view."""
    if df.empty:
        return go.Figure().add_annotation(text="No data to display", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Limit to top items for performance
    df_limited = df.nlargest(max_items, 'bytes') if len(df) > max_items else df
    
    # Get unique labels and their sizes
    label_sizes = {}
    for _, row in df_limited.iterrows():
        prefix = row['prefix']
        if prefix != "(root)":
            label_sizes[prefix] = row['bytes']
            # Also add parent sizes
            parts = prefix.split('/')
            for i in range(len(parts)):
                parent = '/'.join(parts[:i+1])
                if parent not in label_sizes:
                    label_sizes[parent] = 0
                label_sizes[parent] += row['bytes']
    
    if not label_sizes:
        return go.Figure().add_annotation(text="No hierarchical data to display", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Create sunburst
    labels = list(label_sizes.keys())
    values = [label_sizes[label] for label in labels]
    parents = []
    
    for label in labels:
        parts = label.split('/')
        if len(parts) == 1:
            parents.append("")
        else:
            parents.append('/'.join(parts[:-1]))
    
    # Create readable size labels for hover text
    readable_sizes = [bytes_to_readable(value) for value in values]
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Size: %{customdata}<br>Percentage: %{percentParent}<extra></extra>',
        customdata=readable_sizes
    ))
    
    fig.update_layout(
        title=f"Hierarchical Folder View (Top {len(df_limited)} folders)",
        height=500
    )
    
    return fig

# Main App
def main():
    st.title("📁 Folder Size Explorer")
    st.markdown("Upload a Blob Inventory file to analyze folder sizes hierarchically")
    
    # Memory usage indicator
    memory_usage = get_memory_usage()
    if memory_usage > 70:
        st.warning(f"⚠️ Current memory usage: {memory_usage:.1f}%")
    else:
        st.info(f"💾 Current memory usage: {memory_usage:.1f}%")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Inventory File",
        type=['parquet', 'csv'],
        help="Upload a Blob Inventory file in Parquet (preferred) or CSV format"
    )
    
    # Alternative: Read from local file system for very large files
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Or use local file")
    use_local_file = st.sidebar.checkbox(
        "Read file from disk",
        help="For files >2GB, place the file in the app directory and use this option"
    )
    
    local_filename = ""
    if use_local_file:
        local_filename = st.sidebar.text_input(
            "Enter filename",
            placeholder="inventory.parquet",
            help="File should be in the same directory as app.py"
        )
        
        # List available files in current directory
        available_files = []
        for ext in ['.parquet', '.csv']:
            available_files.extend([f for f in os.listdir('.') if f.endswith(ext)])
        
        if available_files:
            st.sidebar.write("**Available files:**")
            for f in available_files[:5]:  # Show first 5 files
                if st.sidebar.button(f"📄 {f}", key=f"file_{f}"):
                    local_filename = f
    
    # Sidebar filters (define before processing)
    st.sidebar.header("🔍 Filters")
    
    include_root = st.sidebar.checkbox(
        "Include (root) level blobs", 
        value=True,
        help="Include blobs that don't have folder prefixes"
    )
    
    max_depth = st.sidebar.slider(
        "Max folder depth", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Maximum number of folder levels to analyze"
    )
    
    min_size_mb = st.sidebar.number_input(
        "Min size filter (MB)", 
        min_value=0.0, 
        value=0.0, 
        step=1.0,
        help="Only show folders larger than this size"
    )
    
    search_prefix = st.sidebar.text_input(
        "Search prefix contains",
        placeholder="e.g., logs",
        help="Filter prefixes that contain this text"
    )
    
    # Determine which file source to use
    file_source = None
    file_info = None
    
    if use_local_file and local_filename:
        try:
            file_info = get_file_info_local(local_filename)
            file_source = "local"
        except FileNotFoundError as e:
            st.sidebar.error(f"❌ {str(e)}")
            return
        except Exception as e:
            st.sidebar.error(f"❌ Error reading local file: {str(e)}")
            return
    elif uploaded_file is not None:
        file_info = get_file_info(uploaded_file)
        file_source = "upload"
    
    if file_source is None:
        st.info("👆 Please upload a Blob Inventory file to get started")
        
        # File size limit information
        st.warning("""
        ### 📁 File Size Limit Configuration
        
        **If you see a 200MB upload limit:**
        
        1. **Create** a `.streamlit/config.toml` file in your project folder
        2. **Add** the following content:
        ```toml
        [server]
        maxUploadSize = 2048  # 2GB limit
        ```
        3. **Restart** the Streamlit app
        
        **Alternative for very large files:**
        - Place your file in the same directory as `app.py`
        - The app can be modified to read files directly from disk
        """)
        
        st.markdown("""
        ### Expected File Format
        Your file should contain at minimum:
        - **Name**: Full blob path (e.g., `folder/subfolder/file.txt`)
        - **Content-Length** or **Content-Length**: File size in bytes
        
        Optional columns:
        - **LastModified** or **Last-Modified**: When the blob was last modified
        - **BlobType** or **Blob-Type**: Type of blob (BlockBlob, etc.)
        - **Container**: Container name
        
        💡 **Performance Tips for Large Files (1GB+)**:
        - **Parquet format is strongly recommended** (5-10x faster processing)
        - CSV files will be processed in chunks but may take longer
        - Ensure you have sufficient RAM (recommended: 4GB+ for 1GB files)
        - Close other memory-intensive applications before processing
        """)
        return
    
    # File information
    if file_info:
        st.sidebar.info(f"""
        **File Info:**
        - Source: {"Local" if file_source == "local" else "Upload"}
        - Size: {file_info['size_readable']}
        - Type: {file_info['type'].upper()}
        - Est. rows: ~{file_info['estimated_rows']:,}
        """)
        
        # Warning for large CSV files
        if file_info['type'] == 'csv' and file_info['size_bytes'] > 100 * 1024 * 1024:  # 100MB
            st.sidebar.warning("⚠️ Large CSV file detected. Consider converting to Parquet for better performance.")
    
        # Processing controls
        st.sidebar.header("⚙️ Processing")
        
        if st.sidebar.button("🚀 Process File", type="primary"):
            # Clear any previous results
            if 'agg_df' in st.session_state:
                del st.session_state['agg_df']
            
            start_time = time.time()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                with status_placeholder:
                    st.info(f"🔄 Processing {file_info['size_readable']} {file_info['type'].upper()} file...")
                
                # Process file based on source and type
                if file_source == "local":
                    if file_info['type'] == 'parquet':
                        aggregated_data = process_local_parquet_in_batches(
                            local_filename, max_depth, include_root, progress_placeholder
                        )
                    else:
                        aggregated_data = process_local_csv_in_chunks(
                            local_filename, max_depth, include_root, progress_placeholder
                        )
                else:  # uploaded file
                    if file_info['type'] == 'parquet':
                        aggregated_data = process_parquet_in_batches(
                            uploaded_file, max_depth, include_root, progress_placeholder
                        )
                    else:
                        aggregated_data = process_csv_in_chunks(
                            uploaded_file, max_depth, include_root, progress_placeholder
                        )
                
                # Convert to DataFrame
                agg_df = convert_aggregated_to_dataframe(aggregated_data)
                
                # Store in session state
                st.session_state['agg_df'] = agg_df
                st.session_state['processing_time'] = time.time() - start_time
                
                progress_placeholder.empty()
                status_placeholder.success(f"✅ Processing completed in {time.time() - start_time:.2f}s")
                
                # Force garbage collection
                del aggregated_data
                gc.collect()
                
            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.error(f"❌ Error processing file: {str(e)}")
                return
    else:
        st.info("👆 Please upload a file or select a local file to analyze")
    
    # Display results if available
    if 'agg_df' in st.session_state:
        agg_df = st.session_state['agg_df'].copy()
        processing_time = st.session_state.get('processing_time', 0)
        
        st.success(f"✅ File processed in {processing_time:.2f}s")
        
        # Apply filters to the processed data
        if min_size_mb > 0:
            agg_df = agg_df[agg_df['mb'] >= min_size_mb]
        
        if search_prefix:
            agg_df = agg_df[agg_df['prefix'].str.contains(search_prefix, case=False, na=False)]
        
        if agg_df.empty:
            st.warning("No data matches your current filters. Try adjusting the settings.")
            return
        
        # Main content area - Fixed total size calculation
        col1, col2, col3, col4 = st.columns(4)
        
        # Add explanation above metrics
        st.info("💡 **Size Calculation**: 'Root Level Total' shows size of top-level folders only (no double counting). 'All Levels Total' includes nested folder sizes.")
        
        with col1:
            # Calculate root level total (level 1 folders only) - this avoids double counting
            root_level_total = agg_df[agg_df['level'] == 1]['bytes'].sum()
            st.metric("Root Level Total", bytes_to_readable(root_level_total))
            
            # Show the inflated total as well for reference
            all_levels_total = agg_df.groupby('prefix')['bytes'].first().sum()
            st.caption(f"All Levels: {bytes_to_readable(all_levels_total)}")
        
        with col2:
            # File count from root level only
            root_level_files = agg_df[agg_df['level'] == 1]['file_count'].sum()
            st.metric("Total Files", f"{root_level_files:,}")
        
        with col3:
            unique_prefixes = agg_df['prefix'].nunique()
            st.metric("Unique Prefixes", f"{unique_prefixes:,}")
        
        with col4:
            avg_file_size = root_level_total / max(root_level_files, 1)
            st.metric("Avg File Size", bytes_to_readable(avg_file_size))
        
        # Visualizations
        st.header("📊 Visualizations")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔥 Treemap", "☀️ Sunburst", "🌳 Folder Tree"])
        
        with viz_tab1:
            st.info("💡 Shows top 30 largest folders with cleaner hierarchical grouping")
            with st.spinner("Creating treemap..."):
                treemap_fig = create_treemap(agg_df, max_items=30)
                st.plotly_chart(treemap_fig, use_container_width=True)
        
        with viz_tab2:
            st.info("💡 Interactive hierarchical view - click to drill down")
            with st.spinner("Creating sunburst..."):
                sunburst_fig = create_sunburst(agg_df, max_items=200)
                st.plotly_chart(sunburst_fig, use_container_width=True)
        
        with viz_tab3:
            st.info("💡 File system tree view showing folder hierarchy with sizes")
            with st.spinner("Building folder tree..."):
                tree_html = create_hierarchy_tree(agg_df)
                
                # Display in a scrollable container
                st.markdown(f"""
                <div style="height: 500px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #dee2e6; font-family: monospace;">
                    {tree_html}
                </div>
                """, unsafe_allow_html=True)
        
        # Top folders card
        st.header("🔝 Top Folders by Size")
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Top 10 Largest Folders")
            top_folders = agg_df.groupby('prefix').agg({
                'bytes': 'first',
                'file_count': 'first',
                'level': 'first'
            }).reset_index().nlargest(10, 'bytes')
            
            # Create a nice table format
            for i, (_, row) in enumerate(top_folders.iterrows(), 1):
                folder_name = row['prefix']
                size_str = bytes_to_readable(row['bytes'])
                files_str = f"{row['file_count']:,}"
                
                # Truncate long folder names
                display_name = folder_name if len(folder_name) <= 40 else folder_name[:37] + "..."
                
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;">
                    <strong>#{i}</strong> 📁 <code>{display_name}</code><br>
                    <small>📊 {size_str} • 📄 {files_str} files • Level {row['level']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Quick Stats")
            
            # Level distribution
            level_counts = agg_df['level'].value_counts().sort_index()
            st.markdown("**Folders by Level:**")
            for level, count in level_counts.items():
                st.write(f"📁 Level {level}: {count:,} folders")
            
            # Size distribution
            st.markdown("**Size Distribution:**")
            large_folders = len(agg_df[agg_df['gb'] >= 1])
            medium_folders = len(agg_df[(agg_df['mb'] >= 100) & (agg_df['gb'] < 1)])
            small_folders = len(agg_df[agg_df['mb'] < 100])
            
            st.write(f"🔴 Large (≥1GB): {large_folders:,}")
            st.write(f"🟡 Medium (100MB-1GB): {medium_folders:,}")
            st.write(f"🟢 Small (<100MB): {small_folders:,}")
        
        # Data table
        st.header("📋 Detailed View")
        
        table_tab1, table_tab2 = st.tabs(["📊 By Folder Levels", "📋 Complete Table"])
        
        with table_tab1:
            st.info("💡 Folders organized by depth level - easier to understand hierarchy")
            
            # Create tables by level
            tables_by_level = create_folder_table(agg_df, max_depth)
            
            for level_name, level_df in tables_by_level.items():
                if not level_df.empty:
                    st.subheader(level_name)
                    st.dataframe(level_df, use_container_width=True, hide_index=True)
                    st.markdown("---")
        
        with table_tab2:
            st.info("💡 Complete sortable table with all folder data")
            
            # Show limited rows for performance
            display_limit = 1000
            if len(agg_df) > display_limit:
                st.info(f"📊 Showing top {display_limit:,} rows by size (out of {len(agg_df):,} total). Use filters to narrow down results.")
                display_df = agg_df.nlargest(display_limit, 'bytes').copy()
            else:
                display_df = agg_df.copy()
            
            # Prepare display dataframe
            display_df['Size'] = display_df['bytes'].apply(bytes_to_readable)
            display_df['Files'] = display_df['file_count'].apply(lambda x: f"{x:,}")
            display_df = display_df[['level', 'prefix', 'Size', 'Files', 'bytes']].rename(columns={
                'level': 'Level',
                'prefix': 'Prefix',
                'bytes': 'Bytes'
            })
            
            # Configure AgGrid
            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum')
            gb.configure_column("Bytes", hide=True)  # Hide raw bytes column
            gb.configure_column("Size", sort='desc')
            
            grid_options = gb.build()
            
            AgGrid(
                display_df,
                gridOptions=grid_options,
                data_return_mode='FILTERED_AND_SORTED',
                update_mode='NO_UPDATE',
                fit_columns_on_grid_load=True,
                height=400
            )
        
        # Download section
        st.header("💾 Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = agg_df.to_csv(index=False)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name="folder_sizes.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON download
            json_data = agg_df.groupby('prefix')['bytes'].first().to_dict()
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="📄 Download as JSON",
                data=json_str,
                file_name="folder_sizes.json",
                mime="application/json"
            )
        
        # Optional: Two-level quick view
        st.header("🚀 Quick Two-Level View")
        if st.checkbox("Show simplified two-level view"):
            quick_df = agg_df[agg_df['level'] <= 2].copy()
            st.dataframe(
                quick_df[['prefix', 'bytes', 'file_count']].rename(columns={
                    'prefix': 'Folder',
                    'bytes': 'Size (bytes)', 
                    'file_count': 'File Count'
                }),
                use_container_width=True
            )
    
    # Show instructions if no file is processed yet
    if 'agg_df' not in st.session_state and file_source is None:
        st.info("👆 Click 'Process File' to analyze your inventory file")

if __name__ == "__main__":
    main()