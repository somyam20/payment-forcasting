# Architecture Documentation

## System Overview

The Meeting Document Generator is a modular, AI-powered application that transforms video meeting recordings into structured documentation. The system follows a layered architecture with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Streamlit Frontend (src/frontend/streamlit_app.py)          │   │
│  │  • User interface and interaction                            │   │
│  │  • Session state management                                  │   │
│  │  • File upload and download                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Document Generation (src/document/)                         │   │
│  │  • DocumentGenerator: Main document creation engine          │   │
│  │  • MermaidDiagramGenerator: Diagram rendering                │   │
│  │  • MermaidEditor: Interactive diagram editing                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   Video      │     │    Audio     │     │   Parallel   │         │
│  │  Processing  │     │  Processing  │     │  Processing  │         │
│  └──────────────┘     └──────────────┘     └──────────────┘         │
│  • VideoProcessor │  • WhisperProcessor│  • Parallel execution      │
│  • Screenshot     │  • SpeechProcessor │  • Chunk processing        │
│    Extractor      │  • Azure clients   │  • Multi-threading         │
│  • Frame analysis │  • Audio validation│                            │
│  • Scene detection│                    │                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         UTILITY LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Core Utilities (src/utils/)                                 │   │
│  │  • Configuration management                                  │   │
│  │  • Logging and audit                                         │   │
│  │  • Cost tracking                                             │   │
│  │  • Media utilities                                           │   │
│  │  • OpenAI integration                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Frontend Layer

**Location**: `src/frontend/streamlit_app.py`

**Responsibilities**:
- User interface rendering using Streamlit
- File upload/download handling
- Session state management
- User interaction and feedback
- Integration with processing components

**Key Components**:
- `main()`: Entry point for the Streamlit application
- Session state management for video, screenshots, and documents
- UI components for video upload, processing controls, and document generation

### 2. Document Generation Layer

**Location**: `src/document/`

**Components**:

#### DocumentGenerator (`document_generator.py`)
- **Purpose**: Main engine for creating PDF and DOCX documents
- **Key Methods**:
  - `generate_docx()`: Creates Word documents
  - `generate_pdf()`: Creates PDF documents
  - `get_document_bytes()`: Returns document as bytes
  - `_generate_narrative_documentation()`: AI-powered content generation
  - `_generate_process_map()`: Creates process flow diagrams

#### MermaidDiagramGenerator (`mermaid_integration.py`)
- **Purpose**: Renders Mermaid diagrams using multiple fallback methods
- **Rendering Methods**:
  1. Graphviz (local)
  2. Mermaid CLI
  3. Mermaid.ink API
  4. Kroki.io API
  5. PlantUML (fallback)

#### MermaidEditor (`mermaid_editor.py`)
- **Purpose**: Interactive diagram editing interface
- **Features**: Live preview, syntax validation, approval workflow

### 3. Processing Layer

#### Video Processing (`src/processors/video/`)

**VideoProcessor** (`video_processor.py`):
- Frame extraction and analysis
- Video metadata extraction
- Frame-by-frame processing

**ScreenshotExtractor** (`screenshot_extractor.py`):
- **Detection Methods**:
  1. Speech keyword triggers
  2. Mouse cursor tracking
  3. Scene change detection (SSIM)
  4. AI-powered content analysis
  5. Text change detection
- **Features**:
  - Parallel processing support
  - Deduplication logic
  - Face blurring (PII protection)
  - Cooldown periods to prevent duplicates

#### Audio Processing (`src/processors/audio/`)

**WhisperProcessor** (`whisper_processor.py`):
- Local Whisper model integration
- Model optimization and caching
- Batch processing

**SpeechProcessor** (`speech_processor.py`):
- Speech recognition using multiple engines
- Keyword detection
- Audio chunk processing

**Azure Clients**:
- `azure_whisper_client.py`: Azure OpenAI Whisper integration
- `azure_speech_client.py`: Azure Speech Services integration
- `azure_whisper_service.py`: Service wrapper

**AudioValidator** (`audio_validator.py`):
- Audio file validation
- Format checking
- Quality assessment

#### Parallel Processing (`src/processors/parallel/`)

**Purpose**: Optimize performance through parallel execution

**Key Modules**:
- `direct_parallel.py`: Direct parallel speech processing
- `parallel_speech_processor.py`: Parallel speech recognition
- `parallel_speech_processing.py`: Speech processing pipeline
- `parallel_speech.py`: Speech chunk processing
- `parallel_video_processor.py`: Video frame processing
- `parallel_whisper_processor.py`: Whisper transcription parallelization

### 4. Utility Layer

**Location**: `src/utils/`

**Key Utilities**:

#### Configuration (`config_loader.py`)
- YAML configuration file loading
- Environment variable substitution
- Centralized configuration management

#### Logging (`logger_config.py`, `audit_logger.py`)
- Application logging setup
- Audit trail logging to CSV
- Usage cost logging

#### Media Utilities (`media_utils.py`)
- Video information extraction
- Timestamp formatting
- Whisper transcription wrapper
- Speech transcript analysis

#### OpenAI Integration (`openai_config.py`, `openai_analyzer.py`)
- OpenAI/Azure OpenAI client management
- Speech analysis for screenshot moments
- Content generation

#### Cost Tracking (`cost_logger.py`, `api_usage_logger.py`)
- API usage logging
- Cost calculation and tracking
- Usage aggregation

## Data Flow

### Video Processing Flow

```
1. User uploads video
   ↓
2. VideoProcessor extracts metadata
   ↓
3. ScreenshotExtractor processes video:
   a. Extract audio → Audio chunks
   b. Process frames → Screenshot candidates
   c. Speech recognition → Timestamps
   d. AI analysis → Key moments
   ↓
4. Deduplication and filtering
   ↓
5. Final screenshots with timestamps
```

### Document Generation Flow

```
1. User selects document type and options
   ↓
2. DocumentGenerator initialized with:
   - Screenshots
   - Speech segments
   - Document type
   ↓
3. AI generates narrative structure:
   - Sections
   - Content
   - Screenshot assignments
   ↓
4. Process map generation (if enabled):
   - Mermaid code generation
   - Diagram rendering
   ↓
5. Document assembly:
   - Add sections
   - Insert screenshots
   - Add diagrams
   - Format content
   ↓
6. Export to PDF/DOCX
```

## Configuration Management

### Configuration Files (`config/`)

- **app_config.yaml**: Application settings, storage paths
- **model_config.yaml**: LLM model configurations
- **whisper_config.yaml**: Whisper model settings
- **logging_config.yaml**: Logging configuration

### Environment Variables

Loaded from `.env` file:
- Azure OpenAI credentials
- OpenAI API keys
- Storage paths
- Service endpoints

## Error Handling Strategy

1. **Graceful Degradation**: System continues with available features if some fail
2. **Multiple Fallbacks**: Each component has fallback mechanisms
3. **Comprehensive Logging**: All errors logged with context
4. **User Feedback**: Clear error messages in UI

## Performance Optimizations

1. **Parallel Processing**: Multi-threaded execution for CPU-intensive tasks
2. **Caching**: Model caching, result caching
3. **Chunking**: Large files processed in chunks
4. **Lazy Loading**: Components loaded only when needed
5. **Resource Management**: Proper cleanup of temporary files

## Security Considerations

1. **PII Protection**: Face blurring in screenshots
2. **Secure Configuration**: Environment variables for secrets
3. **Input Validation**: File type and size validation
4. **Audit Logging**: Track all user actions
5. **Session Management**: Secure session handling

## Scalability

### Current Architecture Supports:
- Single-user processing
- Local file processing
- Configurable parallel workers

### Future Scalability Options:
- Distributed processing (message queues)
- Cloud storage integration
- Multi-user support
- API-based architecture

## Technology Stack

- **Frontend**: Streamlit
- **Video Processing**: OpenCV, FFmpeg
- **Audio Processing**: Whisper, Azure Speech Services
- **AI/ML**: OpenAI API, Azure OpenAI
- **Document Generation**: python-docx, ReportLab
- **Diagram Rendering**: Graphviz, Mermaid
- **Configuration**: PyYAML
- **Logging**: Python logging, CSV files

