# File Explanation and Relations

This document explains the purpose of each file and how they relate to each other in the Meeting Document Generator codebase.

## Project Structure

```
meeting-document-generator/
├── app.py                          # Entry point
├── config/                         # Configuration files
├── src/                            # Source code
│   ├── frontend/                   # User interface
│   ├── document/                   # Document generation
│   ├── processors/                 # Processing modules
│   │   ├── video/                  # Video processing
│   │   ├── audio/                  # Audio processing
│   │   ├── parallel/               # Parallel processing
│   │   └── utils/                  # Processor utilities
│   └── utils/                      # Core utilities
├── data/                           # Data storage
├── tests/                          # Test files
└── requirements.txt                # Dependencies
```

## Entry Point

### `app.py`
**Purpose**: Application entry point  
**Relations**:
- Imports `main()` from `src/frontend/streamlit_app.py`
- Sets up Python path for imports
- Runs Streamlit application

**Dependencies**: None (top-level entry)

---

## Frontend Layer

### `src/frontend/streamlit_app.py`
**Purpose**: Main Streamlit user interface  
**Key Functions**:
- `main()`: Application entry point
- `reset_session()`: Clears session state
- `delete_screenshot()`: Removes screenshot from list

**Imports From**:
- `src.processors.video.video_processor`: VideoProcessor
- `src.processors.video.screenshot_extractor`: ScreenshotExtractor
- `src.utils.media_utils`: Video utilities
- `src.document.document_generator`: DocumentGenerator
- `src.utils.audit_logger`: Logger
- `src.utils.cost_logger`: UsageCostLogger
- `src.utils.config_loader`: Configuration

**Relations**:
- **Uses**: All processing and utility modules
- **Called By**: `app.py`

---

## Document Generation Layer

### `src/document/document_generator.py`
**Purpose**: Core document generation engine  
**Key Classes**:
- `MermaidDiagramGenerator`: Renders Mermaid diagrams
- `MermaidDiagramGenerator_v1`: Legacy diagram generator
- `DocumentGenerator`: Main document creation class

**Key Methods**:
- `generate_docx()`: Creates Word documents
- `generate_pdf()`: Creates PDF documents
- `get_document_bytes()`: Returns document as bytes
- `_generate_narrative_documentation()`: AI content generation
- `_generate_process_map()`: Process flow diagrams

**Imports From**:
- `src.utils.openai_config`: OpenAI client
- `src.utils.api_usage_logger`: Usage logging
- `src.utils.media_utils`: Timestamp formatting

**Relations**:
- **Used By**: `streamlit_app.py`
- **Uses**: OpenAI services, media utilities

### `src/document/mermaid_integration.py`
**Purpose**: Mermaid diagram integration  
**Relations**:
- **Used By**: `document_generator.py`
- **Uses**: `mermaid_editor.py`

### `src/document/mermaid_editor.py`
**Purpose**: Interactive Mermaid diagram editor  
**Relations**:
- **Used By**: `mermaid_integration.py`, `streamlit_app.py`

---

## Video Processing Layer

### `src/processors/video/video_processor.py`
**Purpose**: Video file processing and frame extraction  
**Key Methods**:
- `get_frame_at_position()`: Extract frame at timestamp
- `get_video_info()`: Extract video metadata

**Relations**:
- **Used By**: `screenshot_extractor.py`
- **Uses**: OpenCV

### `src/processors/video/screenshot_extractor.py`
**Purpose**: Intelligent screenshot extraction from video  
**Key Methods**:
- `extract_audio_from_video()`: Audio extraction
- `two_phase_process()`: Main processing pipeline
- `extract_all_speech_keywords()`: Speech processing
- `detect_scene_changes()`: Scene change detection

**Imports From**:
- `src.processors.video.video_processor`: VideoProcessor
- `src.processors.audio.whisper_processor`: WhisperProcessor
- `src.processors.audio.azure_speech_client`: Azure Speech
- `src.processors.audio.azure_whisper_client`: Azure Whisper
- `src.utils.media_utils`: Transcription utilities
- `src.utils.openai_analyzer`: AI analysis
- `src.processors.utils.face_pii`: Face blurring

**Relations**:
- **Used By**: `streamlit_app.py`
- **Uses**: Video processor, audio processors, utilities

---

## Audio Processing Layer

### `src/processors/audio/whisper_processor.py`
**Purpose**: Local Whisper model integration  
**Key Methods**:
- `transcribe_audio()`: Transcribe audio file
- `transcribe_audio_segment()`: Transcribe audio segment

**Relations**:
- **Used By**: `screenshot_extractor.py`, `media_utils.py`
- **Uses**: OpenAI Whisper library

### `src/processors/audio/speech_processor.py`
**Purpose**: Speech recognition processing  
**Key Methods**:
- `process_audio_chunk()`: Process single audio chunk
- `process_audio_chunks_parallel()`: Parallel processing

**Imports From**:
- `src.utils.media_utils`: Transcription utilities
- `src.utils.openai_analyzer`: OpenAI availability

**Relations**:
- **Used By**: Parallel processing modules
- **Uses**: SpeechRecognition library, media utilities

### `src/processors/audio/azure_whisper_client.py`
**Purpose**: Azure OpenAI Whisper API client  
**Key Methods**:
- `transcribe_audio_segment()`: Azure Whisper transcription
- `is_available()`: Check service availability

**Relations**:
- **Used By**: `screenshot_extractor.py`, parallel processors
- **Uses**: Azure OpenAI SDK

### `src/processors/audio/azure_speech_client.py`
**Purpose**: Azure Speech Services client  
**Relations**:
- **Used By**: `screenshot_extractor.py`, parallel processors
- **Uses**: Azure Cognitive Services Speech SDK

### `src/processors/audio/azure_whisper_service.py`
**Purpose**: Azure Whisper service wrapper  
**Relations**:
- **Used By**: Azure clients
- **Uses**: Azure OpenAI SDK

### `src/processors/audio/audio_validator.py`
**Purpose**: Audio file validation  
**Relations**:
- **Used By**: Audio processing pipeline
- **Uses**: Audio processing libraries

### `src/processors/audio/whisper_optimization.py`
**Purpose**: Whisper model optimization and caching  
**Relations**:
- **Used By**: `whisper_processor.py`
- **Uses**: Whisper library

---

## Parallel Processing Layer

### `src/processors/parallel/direct_parallel.py`
**Purpose**: Direct parallel speech processing  
**Key Functions**:
- `process_speech_chunk()`: Process chunk in parallel

**Imports From**:
- `src.utils.media_utils`: Transcription
- `src.utils.logger_config`: Logging

**Relations**:
- **Used By**: `screenshot_extractor.py`
- **Uses**: Media utilities, concurrent.futures

### `src/processors/parallel/parallel_speech_processor.py`
**Purpose**: Parallel speech recognition processor  
**Relations**:
- **Used By**: Parallel processing pipeline
- **Uses**: Media utilities, speech recognition

### `src/processors/parallel/parallel_speech_processing.py`
**Purpose**: Speech processing pipeline  
**Relations**:
- **Used By**: Speech processing workflows
- **Uses**: Media utilities, OpenAI analyzer

### `src/processors/parallel/parallel_speech.py`
**Purpose**: Parallel speech chunk processing  
**Relations**:
- **Used By**: Parallel processing workflows
- **Uses**: Media utilities

### `src/processors/parallel/parallel_video_processor.py`
**Purpose**: Parallel video frame processing  
**Relations**:
- **Used By**: Video processing workflows
- **Uses**: Video processor

### `src/processors/parallel/parallel_whisper_processor.py`
**Purpose**: Parallel Whisper transcription  
**Relations**:
- **Used By**: Audio processing workflows
- **Uses**: Whisper processor

### `src/processors/parallel/parallel_extractor.py`
**Purpose**: Parallel extraction utilities  
**Relations**:
- **Used By**: Extraction workflows

### `src/processors/parallel/parallel_processor.py`
**Purpose**: General parallel processing utilities  
**Relations**:
- **Used By**: Various processing modules

---

## Processor Utilities

### `src/processors/utils/chunk_processor.py`
**Purpose**: Chunk processing utilities  
**Relations**:
- **Used By**: Processing modules

### `src/processors/utils/face_pii.py`
**Purpose**: Face detection and blurring for PII protection  
**Relations**:
- **Used By**: `screenshot_extractor.py`
- **Uses**: Face detection libraries

---

## Core Utilities

### `src/utils/config_loader.py`
**Purpose**: YAML configuration file loader  
**Key Class**: `ConfigLoader`
**Key Methods**:
- `get_config()`: Load configuration file
- `_substitute_env_vars()`: Replace environment variables

**Relations**:
- **Used By**: All modules that need configuration
- **Uses**: PyYAML

### `src/utils/media_utils.py`
**Purpose**: Media processing utilities  
**Key Functions**:
- `transcribe_with_whisper()`: Whisper transcription wrapper
- `analyze_speech_transcript()`: Speech analysis
- `get_video_info()`: Video metadata
- `format_timestamp()`: Timestamp formatting

**Imports From**:
- `src.processors.audio.whisper_processor`: Local Whisper
- `src.utils.api_usage_logger`: Usage logging
- `src.utils.openai_config`: OpenAI client

**Relations**:
- **Used By**: Most processing modules
- **Uses**: Whisper processor, OpenAI, usage logger

### `src/utils/openai_config.py`
**Purpose**: OpenAI/Azure OpenAI client configuration  
**Key Functions**:
- `get_openai_client()`: Get configured client
- `get_chat_model_name()`: Get model name

**Relations**:
- **Used By**: All modules using OpenAI
- **Uses**: OpenAI SDK, Azure OpenAI SDK

### `src/utils/openai_analyzer.py`
**Purpose**: OpenAI-powered speech analysis  
**Key Functions**:
- `analyze_speech_for_screenshot_moments()`: Analyze speech segments

**Imports From**:
- `src.utils.media_utils`: Speech analysis
- `src.utils.openai_config`: OpenAI client

**Relations**:
- **Used By**: `screenshot_extractor.py`
- **Uses**: Media utilities, OpenAI config

### `src/utils/logger_config.py`
**Purpose**: Logging configuration setup  
**Key Functions**:
- `setup_logger()`: Configure application logger
- `setup_usage_logger()`: Configure usage logger

**Relations**:
- **Used By**: All modules that need logging
- **Uses**: Python logging

### `src/utils/audit_logger.py`
**Purpose**: Audit trail logging to CSV  
**Key Class**: `Logger`
**Key Methods**:
- `log()`: Log audit event
- `usage_log()`: Log usage event

**Imports From**:
- `src.utils.config_loader`: Configuration

**Relations**:
- **Used By**: `streamlit_app.py`
- **Uses**: Config loader, pandas

### `src/utils/cost_logger.py`
**Purpose**: Usage cost logging to CSV  
**Key Class**: `UsageCostLogger`
**Key Methods**:
- `log()`: Log cost entry

**Imports From**:
- `src.utils.config_loader`: Configuration

**Relations**:
- **Used By**: `streamlit_app.py`
- **Uses**: Config loader, pandas

### `src/utils/api_usage_logger.py`
**Purpose**: API usage logging and aggregation  
**Key Functions**:
- `log_openai_usage()`: Log OpenAI API usage
- `log_whisper_usage()`: Log Whisper usage
- `get_usage_summary()`: Get usage statistics

**Imports From**:
- `src.utils.logger_config`: Logger setup

**Relations**:
- **Used By**: Media utilities, document generator
- **Uses**: Logger config

### `src/utils/usage_cost_extractor.py`
**Purpose**: Extract usage costs from logs  
**Key Functions**:
- `extract_token_usage_from_app_log()`: Parse log file

**Relations**:
- **Used By**: `streamlit_app.py`
- **Uses**: Log parsing

### `src/utils/setup_azure_openai.py`
**Purpose**: Azure OpenAI setup utilities  
**Relations**:
- **Used By**: Setup scripts
- **Uses**: Azure SDK

---

## Configuration Files

### `config/app_config.yaml`
**Purpose**: Application configuration  
**Contains**: Server settings, storage paths, feature flags

### `config/model_config.yaml`
**Purpose**: LLM model configuration  
**Contains**: Model names, deployment names, API versions

### `config/whisper_config.yaml`
**Purpose**: Whisper model configuration  
**Contains**: Model size, device settings, optimization

### `config/logging_config.yaml`
**Purpose**: Logging configuration  
**Contains**: Log levels, formats, file paths

---

## Data Files

### `data/outputs/audit_log.csv`
**Purpose**: Audit trail of user actions  
**Created By**: `audit_logger.py`

### `data/outputs/usage_cost_log.csv`
**Purpose**: API usage cost tracking  
**Created By**: `cost_logger.py`

### `token_aggregate.json`
**Purpose**: Aggregated token usage statistics  
**Created By**: `api_usage_logger.py`

---

## Dependency Graph

```
app.py
  └── streamlit_app.py
       ├── VideoProcessor
       ├── ScreenshotExtractor
       │    ├── VideoProcessor
       │    ├── WhisperProcessor
       │    ├── Azure Speech/Whisper Clients
       │    ├── Media Utils
       │    ├── OpenAI Analyzer
       │    └── Face PII
       ├── DocumentGenerator
       │    ├── OpenAI Config
       │    ├── API Usage Logger
       │    └── Media Utils
       ├── Config Loader
       ├── Audit Logger
       └── Cost Logger
            └── Config Loader
```

## Import Patterns

1. **Frontend** → **Processors** → **Utils**: Top-down dependency
2. **Processors** → **Utils**: Shared utilities
3. **Utils** → **Config**: Configuration access
4. **Document** → **Utils**: Document utilities
5. **Parallel** → **Processors**: Parallel execution of processors

## Key Relationships

1. **ScreenshotExtractor** is the central processing component that orchestrates video, audio, and AI analysis
2. **Media Utils** provides common functions used across multiple modules
3. **Config Loader** centralizes all configuration access
4. **Document Generator** is independent but uses processing results
5. **Parallel processors** wrap sequential processors for performance

