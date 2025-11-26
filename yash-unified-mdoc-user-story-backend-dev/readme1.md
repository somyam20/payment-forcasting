# 📄 Meeting Document Generator

A comprehensive AI-powered application that transforms meeting recordings into professional documentation. The system intelligently extracts key moments from video recordings, transcribes speech, and generates structured documents in multiple formats.

## 🎯 Overview

The Meeting Document Generator automates the process of creating documentation from meeting recordings by:
- **Intelligent Screenshot Extraction**: Captures key moments using multiple detection methods
- **Speech Transcription**: Converts audio to text using Azure Whisper or OpenAI Whisper
- **AI-Enhanced Analysis**: Uses OpenAI/Azure OpenAI to analyze content and generate insights
- **Document Generation**: Creates professional PDF and DOCX documents with multiple templates
- **Cost Tracking**: Monitors and logs usage costs for API calls

## ✨ Features

### Core Capabilities
- 🎬 **Video Processing**: Supports MP4, AVI, MOV, MKV formats
- 📸 **Smart Screenshot Detection**: Multiple detection modes (Basic/Advanced)
  - Speech keyword triggers
  - Mouse cursor tracking
  - Scene change detection
  - AI-powered content analysis
  - Text change detection
- 🎤 **Speech Recognition**: 
  - Azure Whisper integration
  - OpenAI Whisper fallback
  - Parallel processing for performance
- 📝 **Document Types**:
  - Knowledge Transfer Documents
  - Meeting Summaries
  - User Stories
  - General Documentation
- 📊 **Advanced Features**:
  - Process flow diagrams (Mermaid)
  - Missing questions generation
  - PII detection and face blurring
  - Usage cost tracking and logging

## 🏗️ Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Streamlit - app.py)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO UPLOAD & VALIDATION                     │
│  • File upload (MP4, AVI, MOV, MKV)                             │
│  • Video metadata extraction                                    │
│  • Client name input                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PROCESSING MODE SELECTION                      │
│  • Basic Mode: Quick processing                                 │
│  • Advanced Mode: Comprehensive analysis                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO PROCESSING PIPELINE                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  VideoProcessor (video_processor.py)                     │    │
│  │  • Frame extraction                                      │    │
│  │  • Video metadata (FPS, duration, resolution)            │    │
│  └────────────────────┬─────────────────────────────────────┘    │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ScreenshotExtractor (screenshot_extractor.py)           │  │
│  │  • Two-phase processing                                  │  │
│  │  • Multiple detection methods                           │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│        ┌──────────────┼──────────────┐                          │
│        ▼              ▼              ▼                          │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐                   │
│  │  Audio  │  │  Visual  │  │     AI      │                   │
│  │ Extract │  │ Detection│  │  Analysis   │                   │
│  └────┬────┘  └────┬─────┘  └──────┬──────┘                   │
│       │            │                │                           │
│       ▼            ▼                ▼                           │
│  ┌──────────────────────────────────────────┐                  │
│  │  Speech Processing                       │                  │
│  │  • Azure Whisper (azure_whisper_client)   │                  │
│  │  • OpenAI Whisper (fallback)              │                  │
│  │  • Keyword detection                      │                  │
│  │  • AI speech analysis (openai_analyzer)   │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
│  ┌──────────────────────────────────────────┐                  │
│  │  Visual Detection                        │                  │
│  │  • Mouse cursor tracking                 │                  │
│  │  • Scene change detection                │                  │
│  │  • Text change detection (OCR)           │                  │
│  │  • Structural similarity (SSIM)          │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
│  ┌──────────────────────────────────────────┐                  │
│  │  PII Protection                          │                  │
│  │  • Face detection and blurring           │                  │
│  │  (face_pii.py)                           │                  │
│  └──────────────────────────────────────────┘                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCREENSHOT DEDUPLICATION                     │
│  • Timestamp-based grouping                                     │
│  • Priority-based selection                                     │
│  • Cooldown period enforcement                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT GENERATION                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DocumentGenerator (document_generator.py)              │  │
│  │                                                          │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Content Generation                              │   │  │
│  │  │  • AI-powered descriptions (OpenAI/Azure)        │   │  │
│  │  │  • Process flow diagrams (Mermaid)               │   │  │
│  │  │  • Missing questions generation                  │   │  │
│  │  │  • Document type-specific templates              │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                                                          │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Format Export                                   │   │  │
│  │  │  • PDF generation (reportlab)                     │   │  │
│  │  │  • DOCX generation (python-docx)                 │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    USAGE TRACKING & LOGGING                      │
│  • Token usage extraction (usage_cost_extractor)                │
│  • Cost logging (usage_log.py)                                  │
│  • Audit logging (logger.py)                                   │
│  • CSV storage (local_storage/)                                │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
meeting-document-generator/
│
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Core Processing Modules
│   ├── video_processor.py          # Video frame extraction and processing
│   ├── screenshot_extractor.py    # Main screenshot detection logic
│   ├── document_generator.py       # Document generation (PDF/DOCX)
│   └── audio_validator.py          # Audio validation utilities
│
├── Speech Processing
│   ├── whisper_processor.py        # Local Whisper model processor
│   ├── azure_whisper_client.py     # Azure Whisper API client
│   ├── azure_whisper_service.py    # Azure Whisper service wrapper
│   ├── azure_speech_client.py      # Azure Speech Services client
│   ├── speech_processor.py         # Speech processing utilities
│   └── openai_analyzer.py          # OpenAI-based speech analysis
│
├── Parallel Processing
│   ├── parallel_processor.py       # Base parallel processing framework
│   ├── parallel_video_processor.py # Parallel video frame processing
│   ├── parallel_extractor.py       # Parallel screenshot extraction
│   ├── parallel_speech.py          # Parallel speech processing
│   ├── parallel_speech_processor.py # Advanced speech processing
│   ├── parallel_whisper_processor.py # Parallel Whisper processing
│   ├── parallel_video_chunks.py   # Video chunk processing
│   ├── chunk_processor.py          # Chunk processing utilities
│   └── direct_parallel.py          # Direct parallel execution
│
├── AI & Configuration
│   ├── openai_config.py            # OpenAI/Azure OpenAI configuration
│   ├── setup_azure_openai.py       # Azure OpenAI setup utilities
│   └── openai_analyzer.py          # AI-powered content analysis
│
├── Utilities
│   ├── my_utils.py                 # General utility functions
│   ├── face_pii.py                 # Face detection and PII protection
│   ├── mermaid_integration.py      # Mermaid diagram integration
│   ├── mermaid_editor.py           # Mermaid diagram editor
│   └── whisper_optimization.py     # Whisper model optimization
│
├── Logging & Tracking
│   ├── logger.py                   # Audit logging
│   ├── logger_config.py            # Logging configuration
│   ├── usage_log.py                # Usage cost logging
│   ├── usage_logger.py             # Usage logger utilities
│   └── usage_cost_extractor.py     # Token usage extraction
│
├── Storage
│   └── local_storage/              # Local data storage
│       ├── audit_log.csv           # Audit trail
│       └── usage_cost_log.csv      # Usage cost records
│
└── Configuration Files
    ├── startup.sh                  # Startup script
    └── .env                        # Environment variables (not in repo)
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed and in PATH
- Tesseract OCR (for text detection)
- Azure account (for Azure services) or OpenAI API key

### Setup Steps

1. **Clone the repository**
   ```bash
   cd meeting-document-generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies**
   - **FFmpeg**: Required for video/audio processing
     ```bash
     # macOS
     brew install ffmpeg
     
     # Ubuntu/Debian
     sudo apt-get install ffmpeg
     
     # Windows
     # Download from https://ffmpeg.org/download.html
     ```
   
   - **Tesseract OCR**: Required for text detection
     ```bash
     # macOS
     brew install tesseract
     
     # Ubuntu/Debian
     sudo apt-get install tesseract-ocr
     
     # Windows
     # Download from https://github.com/UB-Mannheim/tesseract/wiki
     ```

5. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Azure OpenAI Configuration (Primary)
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_API_KEY=your_azure_api_key
   AZURE_OPENAI_API_VERSION=2024-02-01
   AZURE_GPT_DEPLOYMENT_NAME=your_deployment_name
   
   # OpenAI Configuration (Fallback)
   OPENAI_API_KEY=your_openai_api_key
   
   # Azure Whisper Configuration
   AZURE_WHISPER_CLIENT_COST=0.006  # Cost per minute
   
   # Azure Key Vault (Optional)
   KEY_VAULT_URL=your_key_vault_url
   
   # Storage Configuration
   LOCAL_STORAGE_DIR=local_storage
   USAGE_COST_BLOB_NAME=usage_cost_log.csv
   
   # Application Configuration
   BASE_URL=your_base_url  # For authentication
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```
   
   Or use the startup script:
   ```bash
   chmod +x startup.sh
   ./startup.sh
   ```

## 🚀 Usage

### Basic Workflow

1. **Upload Video**
   - Click "Upload Meeting Recording"
   - Select a video file (MP4, AVI, MOV, MKV)
   - Wait for upload to complete

2. **Enter Client Information**
   - Enter client name in the sidebar
   - This is required before processing

3. **Select Processing Mode**
   - **Basic**: Quick processing with essential features
   - **Advanced**: Comprehensive analysis with all detection methods

4. **Start Analysis**
   - Click "🚀 Start Analysis"
   - The system will:
     - Extract audio from video
     - Process frames for screenshot detection
     - Transcribe speech
     - Analyze content with AI
     - Generate screenshots at key moments

5. **Generate Documents**
   - Select document type:
     - 📚 Knowledge Transfer
     - 📝 Meeting Summary
     - 📖 User Stories
     - 📄 General Documentation
   - Choose format: PDF, DOCX, or Both
   - Configure advanced options:
     - Include Missing Questions
     - Include Process Maps
     - Include Screenshots
   - Click "🚀 Generate"

6. **Download Documents**
   - Navigate to "📄 Downloads" tab
   - Click download buttons for PDF/DOCX files

### Advanced Features

#### Screenshot Detection Methods

The system uses multiple methods to identify key moments:

1. **Speech Keyword Triggers**: Detects important keywords in speech
2. **Mouse Tracking**: Captures frames when mouse interactions occur
3. **Scene Change Detection**: Identifies significant visual changes
4. **Text Change Detection**: Uses OCR to detect text modifications
5. **AI Analysis**: Uses OpenAI to analyze speech and identify important moments

#### Document Types

- **Knowledge Transfer**: Step-by-step instructions with visual guides
- **Meeting Summary**: Key discussion points, decisions, and action items
- **User Stories**: Requirements with acceptance criteria
- **General Documentation**: Comprehensive documentation with full content

## 🔐 Configuration

### Azure Key Vault Integration

The application supports Azure Key Vault for secure credential management. Set `KEY_VAULT_URL` in your `.env` file to enable this feature.

### Authentication

The application supports Azure authentication. Set `BASE_URL` in your `.env` file to enable logout functionality.

## 📊 Monitoring & Logging

### Log Files
- `app.log`: Main application log (rotates daily)
- `usage.log`: Usage tracking log (rotates daily)

### CSV Reports
- `local_storage/audit_log.csv`: Audit trail of all operations
- `local_storage/usage_cost_log.csv`: Detailed cost tracking

### Logged Information
- Session IDs
- Client names
- File information
- Processing times
- Token usage
- API costs
- Document generation details

## 🛠️ Key Components

### VideoProcessor
Handles video file operations:
- Frame extraction
- Video metadata (FPS, duration, resolution)
- Frame-by-frame processing

### ScreenshotExtractor
Main screenshot detection engine:
- Two-phase processing (coarse + fine)
- Multiple detection algorithms
- Parallel processing support
- Deduplication logic

### DocumentGenerator
Document creation engine:
- AI-powered content generation
- Multiple document templates
- PDF and DOCX export
- Mermaid diagram integration
- Screenshot embedding

### WhisperProcessor
Speech transcription:
- Local Whisper model support
- Azure Whisper API integration
- OpenAI Whisper fallback
- Parallel processing

## 🔄 Processing Modes

### Basic Mode
- Speech keyword detection: ✅
- Mouse tracking: ✅
- Scene detection: ❌
- AI analysis: ✅ (if available)
- Faster processing time

### Advanced Mode
- Speech keyword detection: ✅
- Mouse tracking: ✅
- Scene detection: ✅
- AI analysis: ✅ (if available)
- More comprehensive results

## 📝 Dependencies

Key dependencies include:
- `streamlit`: Web interface
- `opencv-python`: Video processing
- `pytesseract`: OCR for text detection
- `whisper`: Speech recognition
- `openai`: AI content generation
- `python-docx`: DOCX document generation
- `reportlab`: PDF document generation
- `azure-cognitiveservices-speech`: Azure Speech Services
- `pillow`: Image processing
- `numpy`: Numerical operations

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and in PATH
   - Verify with: `ffmpeg -version`

2. **Tesseract OCR errors**
   - Install Tesseract OCR
   - Verify with: `tesseract --version`

3. **OpenAI API errors**
   - Check API keys in `.env` file
   - Verify Azure OpenAI configuration
   - Check network connectivity

4. **Memory issues with large videos**
   - Use Basic mode for large files
   - Process videos in chunks
   - Increase system memory

5. **Audio extraction failures**
   - Verify video has audio track
   - Check FFmpeg installation
   - Try different video format

## 📄 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]

## 📞 Support

[Add support contact information here]

---

**Note**: This application processes video and audio data. Ensure you have proper authorization and comply with privacy regulations when processing meeting recordings.
