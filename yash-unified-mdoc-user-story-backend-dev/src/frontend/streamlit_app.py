import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# This must be the first Streamlit command
st.set_page_config(
    page_title="Meeting Summary Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📝 Meeting Summary Generator")
st.markdown("Generate professional meeting summaries with automatic speaker detection")

import os
import tempfile
import zipfile
import io
import traceback
import base64
import uuid
from datetime import datetime

# Import business logic from main.py
from main import (
    process_video,
    generate_document,
    get_video_duration_ffprobe,
    generate_session_id,
    OPENAI_AVAILABLE
)

# Import utilities
from src.utils.media_utils import get_video_info, format_timestamp
from src.utils.config_loader import get_config_loader
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Load configuration
config_loader = get_config_loader()
app_config = config_loader.get_config('app_config.yaml')
BASE_URL = app_config.get('server', {}).get('base_url', os.getenv("BASE_URL", ""))



def get_image_base64(image_path):
    """Get base64 encoded image for display in HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Custom CSS for enhanced UI
# Enhanced Header
st.markdown("""
    <style>
        .header-container {
            margin-top: 2px; /* Reduced top margin for closer positioning */
            padding: 0.5rem; /* Reduced padding for a more compact header */
            text-align: center;
            background: linear-gradient(135deg, #023047 0%, #0077B6 100%);
            border-radius: 10px;
            margin-bottom: 1rem; /* Reduced bottom margin */
            color: white;
        }
        .header-title {
            font-size: 1rem; /* Smaller font size for the title */
            font-weight: 600;
            margin: 0; /* Remove default margins */
        }
        .header-subtitle {
            font-size: 0.8rem; /* Smaller subtitle font size */
            margin: 0.1rem 0 0 0; /* Minimal margins for compactness */
            opacity: 0.85;
            color: #d0d0d0; /* Slightly lighter for contrast */
        }
    </style>
    <div class="header-container">
        <h1 class="header-title">📄 Meeting Documenter</h1>
        <p class="header-subtitle">Transform meeting  recordings into professional documents with AI-enhanced features</p>
    </div>
    """, unsafe_allow_html=True)

# Session state initialization
if 'screenshots' not in st.session_state:
    st.session_state.screenshots = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'current_timestamp' not in st.session_state:
    st.session_state.current_timestamp = 0
if 'speech_text' not in st.session_state:
    st.session_state.speech_text = []
if 'speech_timestamps' not in st.session_state:
    st.session_state.speech_timestamps = []
if 'keyword_results' not in st.session_state:
    st.session_state.keyword_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'video_title' not in st.session_state:
    st.session_state.video_title = ""
if 'video_description' not in st.session_state:
    st.session_state.video_description = ""
if 'client_name' not in st.session_state:
    st.session_state.client_name = None  
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if "session_guid" not in st.session_state:
    st.session_state.session_guid = str(uuid.uuid4())
if "log_uploaded" not in st.session_state:
    st.session_state.log_uploaded = False
if "log_written" not in st.session_state:
    st.session_state.log_written = False





def reset_session():
    """Reset the session state to initial values"""
    st.session_state.screenshots = []
    st.session_state.video_path = None
    st.session_state.video_info = None
    st.session_state.analysis_complete = False
    st.session_state.processing_complete = False
    st.session_state.speech_timestamps = []
    st.session_state.keyword_results = []
    st.session_state.video_title = ""
    st.session_state.video_description = ""
    st.session_state.current_frame = None
    st.session_state.current_timestamp = 0
    st.session_state.speech_text = []
    st.session_state.client_name = None
    st.session_state.session_id = None
    st.session_state.session_guid = None
    st.session_state.log_uploaded = False
    st.session_state.log_written = False
    st.session_state.extractor = None





    if 'auto_download_triggered' in st.session_state:
        del st.session_state.auto_download_triggered
    if 'pdf_document' in st.session_state:
        del st.session_state.pdf_document
    if 'docx_document' in st.session_state:
        del st.session_state.docx_document

def delete_screenshot(index):
    """Delete a screenshot at the given index"""
    st.session_state.screenshots.pop(index)
    st.rerun()

def download_screenshots():
    """Create a zip file with all screenshots"""
    if not st.session_state.screenshots:
        st.error("No screenshots to download.")
        return
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for i, (img, timestamp, reason) in enumerate(st.session_state.screenshots):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            filename = f"screenshot_{i+1:03d}_{format_timestamp(timestamp, for_filename=True)}.png"
            zip_file.writestr(filename, img_byte_arr.getvalue())
            
            # Add a text file with the reason and timestamp
            reason_text = f"Time: {format_timestamp(timestamp)}\nReason: {reason}"
            zip_file.writestr(f"screenshot_{i+1:03d}_info.txt", reason_text)
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    """Main application function"""
    global OPENAI_AVAILABLE

    # Enhanced Header
    # st.markdown("""
    #     <style>
    #         .header-container {
    #             margin-top: 5px; /* Pushes header closer to top */
    #             text-align: center;
    #         }
    #         .header-title {
    #             font-size: 1.5rem; /* Smaller font size */
    #             margin-bottom: 0.2rem;
    #         }
    #         .header-subtitle {
    #             font-size: 1rem; /* Smaller subtitle */
    #             margin-top: 0;
    #             margin-bottom: 0.5rem;
    #             color: #666;
    #         }
    #     </style>
    #     <div class="header-container">
    #         <h1 class="header-title">📄 Meeting Document Generator</h1>
    #         <p class="header-subtitle">Transform recordings into professional documentation with AI-enhanced features</p>
    #     </div>
    #     """, unsafe_allow_html=True)


    # Sidebar with logo and compact design
    with st.sidebar:
        # Logo display - more compact
        try:
            img_str = get_image_base64("yashlogo.png")
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <img src="data:image/png;base64,{img_str}" width="120">
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("### 🎬 MDG")

        # Status indicators
        status_col1, status_col2 = st.columns(2)
        # with status_col1:
        #     if OPENAI_AVAILABLE:
        #         st.markdown('<span class="status-indicator status-success">✓ AI Ready</span>', unsafe_allow_html=True)
        #     else:
        #         st.markdown('<span class="status-indicator status-error">✗ AI Limited</span>', unsafe_allow_html=True)
        
        # with status_col2:
        #     if st.session_state.video_path:
        #         st.markdown('<span class="status-indicator status-success">✓ Video Loaded</span>', unsafe_allow_html=True)
        #     else:
        #         st.markdown('<span class="status-indicator status-warning">○ No Video</span>', unsafe_allow_html=True)

        st.markdown("---")
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button('🚪 Logout', use_container_width=True):
                logout = BASE_URL + '/.auth/logout'
                st.markdown(f'<meta http-equiv="refresh" content="0; url={logout}">', unsafe_allow_html=True)
        
        with btn_col2:
            if st.session_state.get("video_path"):
                if st.button("↪️ Start Over", use_container_width=True):
                    # Session logging is handled in main.py
                    reset_session()

                    st.rerun()

        
        
        if st.session_state.client_name is None and st.session_state.video_path:
            client_name = st.text_input("Enter Client Name:")
            st.session_state.session_guid = str(uuid.uuid4())
            st.session_state.log_uploaded = False



        # Analysis Settings - Compact Card
        if st.session_state.video_path and not st.session_state.analysis_complete:
            st.session_state.session_id = generate_session_id()
            # Session logging is handled in main.py process_video function


            with st.container():
                # st.markdown('<div class="settings-card">', unsafe_allow_html=True)
                # st.markdown('<div class="card-title">⚙️ Processing Mode</div>', unsafe_allow_html=True)

                capture_method = st.radio(
                    "⚙️ **Screenshot Processing Mode**",
                    ["Basic", "Advanced"],
                    help="Basic: Quick processing.\nAdvanced: Quality content with more processing time.",
                    horizontal=True
                )
                
                # Configure detection methods
                if capture_method == "Basic":
                    use_speech = True
                    use_mouse_detection = True
                    use_scene_detection = False
                    use_ai_analysis = bool(OPENAI_AVAILABLE)
                    capture_after_interaction = True
                    trigger_keywords = []
                    detection_mode = 'basic'
                else:
                    use_speech = True
                    use_mouse_detection = True
                    use_scene_detection = True
                    use_ai_analysis = bool(OPENAI_AVAILABLE)
                    capture_after_interaction = True
                    detection_mode = 'advanced'
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Start Analysis Button
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                if not client_name.strip():
                    st.error("❌ Please enter client name before starting analysis.")
                else:
                    st.session_state.client_name = client_name.strip()
                    
                    with st.spinner("Processing video..."):
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(progress_pct, message):
                                progress_bar.progress(progress_pct)
                                status_text.text(message)
                            
                            # Call business logic function from main.py
                            result = process_video(
                                video_path=st.session_state.video_path,
                                client_name=st.session_state.client_name,
                                detection_mode=detection_mode,
                                use_speech=use_speech,
                                use_mouse_detection=use_mouse_detection,
                                use_scene_detection=use_scene_detection,
                                use_ai_analysis=use_ai_analysis,
                                progress_callback=update_progress,
                                session_guid=st.session_state.session_guid
                            )
                            
                            # Store results in session state
                            st.session_state.screenshots = result["screenshots"]
                            st.session_state.speech_timestamps = result["speech_timestamps"]
                            st.session_state.speech_text = result["speech_timestamps"]
                            st.session_state.keyword_results = result.get("keyword_results", [])
                            st.session_state.processing_complete = True
                            st.session_state.analysis_complete = True
                            st.session_state.total_processing_time = result["processing_time"]
                            st.session_state.extractor = result["extractor"]
                            st.session_state.session_id = result["session_id"]
                            st.session_state.session_guid = result["session_guid"]
                            st.session_state.log_uploaded = True
                            
                            progress_bar.progress(1.0)
                            status_text.text("")
                            st.success(f"✅ Processing completed in {result['processing_time']:.1f}s! 💡 Generate documents using the panel on the right →.")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.write(traceback.format_exc())
        
        elif st.session_state.video_path and st.session_state.analysis_complete:
            if st.button("🔄 Reprocess Video", use_container_width=True):
                st.session_state.analysis_complete = False
                st.rerun()

        # Action buttons at bottom
        # st.markdown("---")
        
    # Main content area
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        # Video Upload/Display Section
        if st.session_state.video_path:
            # Video player with enhanced container
            # st.markdown('<div class="video-container video-loaded">', unsafe_allow_html=True)
            st.video(st.session_state.video_path)
            
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Enhanced upload area
            # st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.markdown("#### 🎬 Upload Meeting Recording")
            uploaded_file = st.file_uploader(
                        "Choose a video file",
                        type=["mp4", "avi", "mov", "mkv"],
                        help="Supported: MP4, AVI, MOV, MKV • Max: 2GB",
                        label_visibility="collapsed"
                    )
                    
            if uploaded_file is not None:
                
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                
                # File info with progress
                progress_col1, progress_col2 = st.columns([2, 1])
                with progress_col1:
                    st.info(f"📁 **{uploaded_file.name}** • {file_size_mb:.1f} MB")
                with progress_col2:
                    if file_size_mb > 500:
                        st.warning("⏳ Large file")
                
                reset_session()
                temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                
                with st.spinner(f"Uploading..."):
                    try:
                        with open(temp_file_path, "wb") as f:
                            chunk_size = 8192
                            bytes_data = uploaded_file.getvalue()
                            
                            for i in range(0, len(bytes_data), chunk_size):
                                chunk = bytes_data[i:i + chunk_size]
                                f.write(chunk)
                        
                        st.session_state.video_path = temp_file_path
                        st.session_state.video_info = get_video_info(temp_file_path)
                        st.success("✅ Upload complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Upload failed: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
                

    with main_col2:
        if st.session_state.video_info:
                video_filename = os.path.basename(st.session_state.video_path)
                video_size = os.path.getsize(st.session_state.video_path)
                video_size_mb = round(video_size / (1024 * 1024), 2)
                video_duration = get_video_duration_ffprobe(st.session_state.video_path)

                st.caption(f"📁 **File:** {video_filename}")
                st.caption(f"💾 **Size:** {video_size_mb:.1f} MB")
                if st.session_state.analysis_complete:
                    screenshots_count = len(st.session_state.screenshots)
                    # Logging is now handled in main.py process_video function


    if st.session_state.processing_complete and st.session_state.screenshots:

        # st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        # st.markdown('<div class="card-title">📚 Generate Documents</div>', unsafe_allow_html=True)
        
        # Compact form
        try:
            video_filename = os.path.basename(st.session_state.video_path)
            document_name = video_filename.split('.')[0]
        except:
            document_name = ""
        
        sub_column1, sub_column2 = st.columns([1, 1])

        with sub_column1:
            video_filename = os.path.basename(st.session_state.video_path)
            document_name = video_filename.split('.')[0]
            # Audit logging is now handled in main.py generate_document function
            doc_title = st.text_input("📝 Document Title", value=document_name, key="new_doc_title")

            doc_format = st.selectbox(
                "📄 Select Format",
                options=["PDF", "DOCX", "Both"],
                index=0,
                key="doc_format_select"
            )




        with sub_column2:
            doc_type_options = {
                "User Story Generator": "📝 User Story Generator", 

            }

            selected_doc_type = st.selectbox(
                "📝 Document Type",
                options=list(doc_type_options.keys()),
                format_func=lambda x: doc_type_options[x],
                key="doc_type_selector_new"
            )
            
            st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

            with st.expander("⚙️ Advanced Options", expanded=False):
                enable_missing_questions = st.checkbox("Include Missing Questions", value=True, key="new_missing_questions")
                enable_process_map = st.checkbox("Include Process Maps", value=True, key="new_process_map")
                include_screenshots = st.checkbox("Include Screenshots", value=True, key="screen_shots")

                st.markdown("**Meeting Summary Settings:**")
                doc_format = st.radio("Document Format", ["PDF", "DOCX"], horizontal=True, key="meeting_format")
                
                col1, col2 = st.columns(2)
                with col1:
                    auto_extract_attendees = st.checkbox("Auto-extract Attendees", value=True, key="auto_attendees")
                with col2:
                    auto_extract_highlights = st.checkbox("Auto-extract Highlights", value=True, key="auto_highlights")
            # advanced_options = st.selectbox(
            #     "⚙️ Advanced Options",
            #     options=["Include Missing Questions", "Include Process Maps", "Both"],
            #     index = 0,
            #     key="advanced_options_select"
            # )

    # Generate button
        col1, col2, col3 = st.columns([1, 0.5, 1])  # You can adjust these ratios for finer control

        with col2:
            generate_clicked = st.button("🚀 Generate", type="primary", key="generate_new_doc")

        if generate_clicked:
            if not doc_format:
                st.error("⚠️ Please select at least one format (PDF or DOCX)")
            else:
                with st.spinner("Generating your document..."):
                    try:
                        # Call business logic function from main.py
                        result = generate_document(
                            video_path=st.session_state.video_path,
                            screenshots=st.session_state.screenshots,
                            client_name=st.session_state.client_name,
                            doc_title=doc_title,
                            doc_type=selected_doc_type,
                            doc_format=doc_format,
                            speech_segments=st.session_state.speech_timestamps,
                            enable_missing_questions=enable_missing_questions,
                            enable_process_map=enable_process_map,
                            include_screenshots=include_screenshots,
                            session_guid=st.session_state.session_guid
                        )
                        
                        # Store generated documents in session state
                        if result.get("pdf_bytes"):
                            st.session_state[f'generated_pdf_{selected_doc_type}'] = result["pdf_bytes"]
                        
                        if result.get("docx_bytes"):
                            st.session_state[f'generated_docx_{selected_doc_type}'] = result["docx_bytes"]
                        
                        # Store metadata
                        st.session_state[f'generated_title_{selected_doc_type}'] = result["title"]
                        st.session_state[f'generated_format_{selected_doc_type}'] = result["format"]
                        st.session_state[f'doc_generated_{selected_doc_type}'] = True
                        st.session_state.log_written = True
                        
                        # Show success message
                        st.success("✅ Document is ready for download!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.write(traceback.format_exc())

    # Results section - Enhanced tabs
    if st.session_state.analysis_complete:
        if 'screenshots' in st.session_state and len(st.session_state.screenshots) > 0:
            
            main_tab1, main_tab2 = st.tabs(["📄 Downloads", "🎤 Transcript"])
            
            with main_tab1:
                # Enhanced document downloads table
                doc_type_options = {
                    "User Story Generator": "📝 User Story Generator", 
                }
                
                
                generated_docs = []
                for doc_type in doc_type_options.keys():
                    if st.session_state.get(f'doc_generated_{doc_type}', False):
                        generated_docs.append(doc_type)
                
                if generated_docs:
                    st.markdown("#### 📋 Generated Documents")
                    
                    # Modern table layout
                    for doc_type in generated_docs:
                        doc_name = doc_type_options[doc_type]
                        doc_title = st.session_state.get(f'generated_title_{doc_type}', 'Documentation')
                        generated_format = st.session_state.get(f'generated_format_{doc_type}', 'Both')
                        
                        with st.container():
                            # st.markdown('<div class="doc-item">', unsafe_allow_html=True)
                            
                            item_col1, item_col2, item_col3, item_col4 = st.columns([2, 3, 1, 1])
                            
                            with item_col1:
                                st.markdown(f"**{doc_name}**")
                            with item_col2:
                                st.markdown(f"**{doc_title}**")

                                # st.caption(doc_title)
                            
                            import datetime
                            today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                            
                            with item_col3:
                                if generated_format in ["Both", "PDF"] and f'generated_pdf_{doc_type}' in st.session_state:
                                    st.download_button(
                                        "⬇️PDF",
                                        data=st.session_state[f'generated_pdf_{doc_type}'],
                                        file_name=f"{doc_title.replace(' ', '_')}_{doc_type}_{today_date}.pdf",
                                        mime="application/pdf",
                                        key=f"pdf_download_{doc_type}",
                                        use_container_width=True
                                    )
                            
                            with item_col4:
                                if generated_format in ["Both", "WORD", "DOCX"] and f'generated_docx_{doc_type}' in st.session_state:
                                    st.download_button(
                                        "⬇️DOCX",
                                        data=st.session_state[f'generated_docx_{doc_type}'],
                                        file_name=f"{doc_title.replace(' ', '_')}_{doc_type}_{today_date}.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        key=f"docx_download_{doc_type}",
                                        use_container_width=True
                                    )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("💡 No documents created. Use the generate button to generate documents")
                
                # Legacy support
                if 'pdf_document' in st.session_state and 'docx_document' in st.session_state:
                    st.markdown("### 📄 Legacy Documentation")
                    
                    if 'total_processing_time' in st.session_state:
                        st.success(f"Processing completed in {st.session_state.total_processing_time:.1f}s!")
                    
                    legacy_col1, legacy_col2 = st.columns(2)
                    
                    with legacy_col1:
                        doc_title = st.session_state.get('doc_title', "Documentation")
                        pdf_filename = f"{doc_title.replace(' ', '_')}_documentation.pdf"
                        
                        st.download_button(
                            "📄 Download PDF",
                            data=st.session_state.pdf_document,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            key="pdf_download",
                            use_container_width=True
                        )
                        
                        if 'auto_download_triggered' not in st.session_state:
                            st.info("✅ PDF ready!")
                            st.session_state.auto_download_triggered = True
                    
                    with legacy_col2:
                        docx_filename = f"{doc_title.replace(' ', '_')}_documentation.docx"
                        st.download_button(
                            "📝 Download DOCX",
                            data=st.session_state.docx_document,
                            file_name=docx_filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="docx_download",
                            use_container_width=True
                        )
                        
            with main_tab2:
                # Enhanced transcript display
                if 'speech_text' in st.session_state and len(st.session_state.speech_text) > 0:
                    st.markdown("### 🎤 Speech Recognition Results")
                    
                    # Statistics
                    total_segments = len(st.session_state.speech_text)
                    total_duration = max([ts for ts, text in st.session_state.speech_text]) if st.session_state.speech_text else 0
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Segments", total_segments)
                    with stats_col2:
                        st.metric("Duration", f"{total_duration/60:.1f}m")
                    with stats_col3:
                        total_words = sum(len(text.split()) for ts, text in st.session_state.speech_text)
                        st.metric("Words", total_words)
                    
                    # Expandable full transcript
                    with st.expander("📋 Full Transcript", expanded=False):
                        full_transcript = "\n\n".join([f"**{format_timestamp(ts)}:** {text}" for ts, text in st.session_state.speech_text])
                        st.markdown(full_transcript)
                    
                    # Compact transcript table
                    st.markdown("**Recent Segments:**")
                    
                    # Show last 10 segments
                    recent_segments = st.session_state.speech_text[-10:] if len(st.session_state.speech_text) > 10 else st.session_state.speech_text
                    
                    for ts, text in recent_segments:
                        with st.container():
                            time_col, text_col = st.columns([1, 4])
                            with time_col:
                                st.caption(f"**{format_timestamp(ts)}**")
                            with text_col:
                                st.write(text)
                    
                    if len(st.session_state.speech_text) > 10:
                        st.info(f"Showing recent 10 of {total_segments} segments. View full transcript above.")
                        
                else:
                    # Provide more helpful information
                    if st.session_state.get('analysis_complete', False):
                        st.warning("🔇 No speech recognition data available.")
                        st.info("""
                        **Possible reasons:**
                        - Video has no audio track
                        - Speech recognition failed during processing
                        - Audio extraction encountered an error
                        
                        **To enable speech recognition:**
                        - Ensure your video file contains audio
                        - Check that Azure Whisper or OpenAI Whisper is properly configured
                        - Try reprocessing the video
                        """)
                    else:
                        st.info("🔇 No speech recognition data available. Process a video first to generate speech recognition data.")

if __name__ == "__main__":
    main()