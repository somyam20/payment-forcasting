# Complete List of Changes Required: Meeting Documentation → User Story Generator

This document lists all places in the codebase that need to be modified when transitioning from a "Meeting Documentation" system to a "User Story Generator" system.

---

## 📋 Table of Contents
1. [Frontend/UI Files](#frontendui-files)
2. [Backend API Files](#backend-api-files)
3. [Core Business Logic](#core-business-logic)
4. [Document Generation](#document-generation)
5. [Configuration Files](#configuration-files)
6. [Documentation Files](#documentation-files)
7. [Database/Logging References](#databaselogging-references)
8. [Variable/Function Names](#variablefunction-names)

---

## 1. Frontend/UI Files

### `src/frontend/streamlit_app.py`
**Lines to change:**
- **Line 10**: `page_title="Meeting Summary Generator"` → `page_title="User Story Generator"`
- **Line 11**: `page_icon="📝"` → `page_icon="📖"` (or appropriate icon)
- **Line 16**: `st.title("📝 Meeting Summary Generator")` → `st.title("📖 User Story Generator")`
- **Line 17**: `st.markdown("Generate professional meeting summaries...")` → `st.markdown("Generate user stories from requirements...")`
- **Line 83**: `<h1 class="header-title">📄 Meeting Documenter</h1>` → `<h1 class="header-title">📖 User Story Generator</h1>`
- **Line 84**: `"Transform meeting recordings into professional documents..."` → `"Transform requirements into user stories with acceptance criteria..."`
- **Line 379**: `st.markdown("#### 🎬 Upload Meeting Recording")` → `st.markdown("#### 🎬 Upload Requirements Video/Recording")`
- **Line 469**: `"meeting_summary": "📝 Meeting Summary"` → `"user_story_generator": "📖 User Stories"`
- **Line 473-478**: Update doc_type_options to use `user_story_generator` instead of `meeting_summary`
- **Line 487**: `st.markdown("**Meeting Summary Settings:**")` → `st.markdown("**User Story Settings:**")`
- **Line 492-494**: Remove or update meeting-specific checkboxes (auto-extract attendees, auto-extract highlights)
- **Line 558**: `"meeting_summary": "📝 Meeting Summary"` → `"user_story_generator": "📖 User Stories"`

---

## 2. Backend API Files

### `api.py`
**Lines to change:**
- **Line 3**: `FastAPI application entry point for Meeting Document Generator API` → `FastAPI application entry point for User Story Generator API`
- **Line 19**: `title="MDoc API"` → `title="User Story Generator API"` (or keep MDoc if it's a brand name)
- **Line 21**: `description="Meeting Document Generator API - Transform meeting recordings into professional documents"` → `description="User Story Generator API - Transform requirements into user stories"`
- **Line 47**: `"message": "MDoc API - Meeting Document Generator"` → `"message": "User Story Generator API"`

### `src/backend/routes/document_routes.py`
**Lines to change:**
- **Line 12**: `@router.post("/upload")` - Function name `upload_meeting` → `upload_requirements` (line 13)
- **Line 24**: `Upload and process meeting recording` → `Upload and process requirements video/recording`
- **Line 26**: Comments about meeting recording → requirements video
- **Line 57**: `@router.post("/generate/meeting-summary")` → `@router.post("/generate/user-stories")`
- **Line 58**: Function name `generate_meeting_summary` → `generate_user_stories`
- **Line 69**: `Generate Meeting Summary Document from processed video` → `Generate User Stories Document from processed video`
- **Line 85**: `doc_type="meeting_summary"` → `doc_type="user_story_generator"`

### `src/backend/controllers/document_controller.py`
**Lines to change:**
- **Line 2**: `Document controller for handling meeting document processing` → `Document controller for handling user story generation`
- **Line 36**: `DOCUMENTS_S3_FOLDER = os.getenv("S3_DOCUMENT_FOLDER", "meeting-documents")` → `"user-stories"` or `"documents"`
- **Line 90**: Function name `process_meeting` → `process_requirements` (or keep if generic)
- **Line 101**: `Process uploaded meeting video file` → `Process uploaded requirements video file`
- **Line 235**: `doc_type: str = "meeting_summary"` → `doc_type: str = "user_story_generator"`

---

## 3. Core Business Logic

### `main.py`
**Lines to change:**
- **Line 3**: `Main application file for Meeting Document Generator` → `Main application file for User Story Generator`
- **Line 426, 566, 585**: `tool_name="mDoc_v2"` → Update if needed (or keep if it's a brand name)
- **Line 471**: `doc_type: str = "meeting_summary"` → `doc_type: str = "user_story_generator"`
- **Line 489**: Docstring: `"meeting_summary"` → `"user_story_generator"` in choices
- **Line 516**: `"meeting_summary": "Meeting summary with key discussion points..."` → `"user_story_generator": "Collection of user stories with acceptance criteria..."`
- **Line 608**: `doc_type: str = "meeting_summary"` → `doc_type: str = "user_story_generator"`
- **Line 703**: `parser.add_argument("--doc-type", default="meeting_summary"` → `default="user_story_generator"`
- **Line 704**: `choices=["meeting_summary"]` → `choices=["user_story_generator"]`
- **Line 699**: `description="Meeting Document Generator - CLI Mode"` → `description="User Story Generator - CLI Mode"`

---

## 4. Document Generation

### `src/document/document_generator.py`
**Lines to change:**
- **Line 1264**: `"""Class for generating product documentation from screenshots and video content"""` → Update description
- **Line 1270-1272**: Remove or repurpose `meeting_participants`, `meeting_highlights`, `meeting_duration_minutes` parameters (or rename to user story relevant fields)
- **Line 1284**: `document_type: Type of document to generate ("kt_document" or "meeting_summary")` → `("kt_document" or "user_story_generator")`
- **Line 1297-1299**: Remove or repurpose meeting-specific instance variables:
  - `self.meeting_participants`
  - `self.meeting_highlights`
  - `self.meeting_duration_minutes`
- **Line 1840**: `Generate missing questions that should be asked in future meetings...` → Update to user story context
- **Line 1852**: `Based on the following meeting transcript...` → `Based on the following requirements transcript...`
- **Line 1885**: `"You are an expert meeting analyst..."` → `"You are an expert user story analyst..."`
- **Line 1905**: `"## Missing Questions for Next Meeting\n\n..."` → `"## Missing Requirements\n\n..."` or similar
- **Line 2104**: `if self.document_type == "meeting_summary":` → `if self.document_type == "user_story_generator":`
- **Lines 2105-2115**: Entire meeting metadata section needs to be replaced with user story relevant metadata
- **Line 2118**: `"You are a meeting summary generating agent..."` → `"You are a user story generating agent..."`
- **Line 2120**: `MEETING METADATA:` → `REQUIREMENTS METADATA:` or remove
- **Line 2127**: `This is a MEETING SUMMARY document that should:` → `This is a USER STORY document that should:`
- **Line 2128-2131**: Replace meeting information section with user story format:
  - Date of meeting → Project/Feature name
  - Attendees → Stakeholders/Product Owner
  - Meeting Duration → Not needed
  - Key Discussion Points → Feature Requirements
- **Line 2132-2139**: Update requirements to focus on user stories:
  - Identify user personas
  - Extract user stories with "As a... I want... So that..." format
  - Include acceptance criteria
  - Organize by feature/epic
- **Line 2143**: `"title": "Meeting Summary: [Meeting Topic]"` → `"title": "User Stories: [Feature/Project Name]"`
- **Line 2144**: `"Brief overview of the meeting purpose and participants"` → `"Brief overview of the feature and user stories"`
- **Line 2147**: `"title": "Meeting Information"` → `"title": "Project Information"` or `"Feature Overview"`
- **Line 2148**: Update content template to remove meeting-specific fields
- **Line 2153**: `"title": "Discussion Topics"` → `"title": "User Stories"` or `"Epics"`
- **Line 2167**: `if self.document_type == "meeting_summary":` → `if self.document_type == "user_story_generator":`
- **Line 2169-2170**: `"You are a meeting summary expert..."` → `"You are a user story expert. Your task is to analyze requirements transcripts and create structured user stories with acceptance criteria."`
- **Line 2298**: `"title": "Missing Questions for Next Meeting"` → `"title": "Missing Requirements"` or `"Open Questions"`

---

## 5. Configuration Files

### `config/app_config.yaml`
**Lines to change:**
- **Line 3**: `name: "Meeting Document Generator"` → `name: "User Story Generator"`

### Environment Variables / `.env` (if referenced)
- Update any references to "meeting" in environment variable descriptions or comments

---

## 6. Documentation Files

### `README.md`
**Lines to change:**
- **Line 1**: `# 📄 Meeting Document Generator` → `# 📖 User Story Generator`
- **Line 3**: `An AI-powered application that transforms video meeting recordings...` → `An AI-powered application that transforms requirements into user stories...`
- **Line 7**: `The Meeting Document Generator automates...` → `The User Story Generator automates...`
- **Line 30**: `- Meeting Summaries` → `- User Stories` (in document types list)
- **Line 178**: `Click "Upload Meeting Recording"` → `Click "Upload Requirements Video"`
- **Line 202**: `- 📝 Meeting Summary` → `- 📖 User Stories`
- **Line 440**: `"# MDOC-meeting-summary"` → Update if changing repo name

### `readme1.md`
**Lines to change:**
- **Line 1**: `# 📄 Meeting Document Generator` → `# 📖 User Story Generator`
- **Line 3**: Update description
- **Line 7**: `The Meeting Document Generator automates...` → Update

### `README.Docker.md`
**Lines to change:**
- **Line 1**: `# Docker Setup for Meeting Document Generator` → `# Docker Setup for User Story Generator`
- **Line 3**: Update description

### `architecture.md`
**Lines to change:**
- **Line 1**: `# Architecture Documentation`
- **Line 5**: `The Meeting Document Generator is a modular...` → `The User Story Generator is a modular...`

### `file_explanation_and_relation.md`
**Lines to change:**
- **Line 3**: `This document explains... in the Meeting Document Generator codebase.` → Update

### `Dockerfile`
**Lines to change:**
- **Line 1**: `# Multi-stage build for Meeting Document Generator` → Update

### `app.py`
**Lines to change:**
- **Line 3**: `Entry point for Meeting Document Generator` → Update

---

## 7. Database/Logging References

### `data/outputs/usage_cost_log.csv`
- **Note**: Historical data will contain "Meeting Documenter" - this is fine for historical records
- **Future entries**: Will automatically update once code changes are made (no manual changes needed)

### `data/outputs/audit_log.csv`
- **Note**: Historical data will contain "Meeting Documenter" - this is fine for historical records
- **Future entries**: Will automatically update once code changes are made

### `src/utils/api_usage_logger.py` (if it has hardcoded strings)
- Check for any "Meeting Documenter" or "meeting" references in logging messages

### `src/utils/cost_logger.py` (if it has hardcoded strings)
- Check for any "Meeting Documenter" or "meeting" references

---

## 8. Variable/Function Names

### Key Variables to Consider Renaming:
1. **Function names:**
   - `process_meeting()` → `process_requirements()` or keep generic
   - `generate_meeting_summary()` → `generate_user_stories()`
   - `upload_meeting()` → `upload_requirements()`

2. **Variable names:**
   - `meeting_participants` → `stakeholders` or remove
   - `meeting_highlights` → `feature_requirements` or `key_points`
   - `meeting_duration_minutes` → Remove or repurpose
   - `meeting_summary` (doc_type) → `user_story_generator`

3. **Session state variables** (in `streamlit_app.py`):
   - Any meeting-specific session state variables

---

## 9. Additional Considerations

### S3 Folder Names
- **Line 36 in `document_controller.py`**: `"meeting-documents"` → `"user-stories"` or `"documents"`

### API Endpoints
- `/api/document/generate/meeting-summary` → `/api/document/generate/user-stories`
- Consider backward compatibility if needed

### Test Files
- `test_upload.py` - Check for any meeting-specific references

### Missing Questions Section
- The "Missing Questions for Next Meeting" feature should be repurposed to:
  - "Missing Requirements" or
  - "Open Questions" or
  - "Clarifications Needed"

### Process Maps
- Process maps might still be relevant for user stories (showing user flows)
- Keep or adapt based on requirements

---

## 10. Summary Checklist

- [ ] Update all UI text and titles
- [ ] Change document type from `meeting_summary` to `user_story_generator`
- [ ] Update API endpoints and function names
- [ ] Modify document generation prompts for user stories
- [ ] Remove/repurpose meeting-specific metadata fields
- [ ] Update all documentation files
- [ ] Change S3 folder names
- [ ] Update configuration files
- [ ] Test all document generation flows
- [ ] Verify API endpoints work correctly
- [ ] Update any hardcoded strings in utility files

---

## Notes

1. **Backward Compatibility**: Consider if you need to support both `meeting_summary` and `user_story_generator` document types during transition.

2. **Database Migration**: Historical logs will retain old names - this is typically acceptable.

3. **Brand Names**: If "MDoc" or "mDoc_v2" are brand names, you may want to keep them or update consistently.

4. **User Story Format**: Ensure the AI prompts generate proper user stories in the format:
   - "As a [user type], I want [goal], so that [benefit]"
   - With acceptance criteria
   - Organized by epic/feature

5. **Testing**: After making changes, thoroughly test:
   - Video upload and processing
   - Document generation
   - API endpoints
   - UI workflows

