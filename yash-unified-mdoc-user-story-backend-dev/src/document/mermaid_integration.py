"""
Mermaid Integration Module for Meeting Document Generator

This module handles the integration of the Mermaid editor with the main app workflow,
providing a seamless experience for users to edit and approve diagrams before document generation.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from .mermaid_editor import MermaidEditor, convert_dot_to_mermaid
import logging

from ..utils.logger_config import setup_logger

setup_logger()
def show_diagram_approval_workflow(doc_generator) -> bool:
    """
    Show the interactive Mermaid diagram approval workflow
    
    Args:
        doc_generator: DocumentGenerator instance with pending diagrams
        
    Returns:
        True if user has approved/completed the workflow, False if still editing
    """
    if not hasattr(doc_generator, 'pending_mermaid_codes') or not doc_generator.pending_mermaid_codes:
        return True  # No diagrams to approve, proceed with document generation
    
    st.markdown("## 🎨 Interactive Diagram Editor")
    st.info("📋 We've generated process flow diagrams from your video. Edit the diagrams below and click 'Approve & Use' for each one when you're satisfied.")
    st.warning("⚠️ **Important**: Don't use Ctrl+Enter as it refreshes the page. Use the 'Approve & Use' button instead.")
    
    if 'diagram_approval_state' not in st.session_state:
        st.session_state.diagram_approval_state = {
            'current_step': 'editing',
            'approved_diagrams': [],
            'edited_codes': [],
            'section_titles': []
        }
    
    approval_state = st.session_state.diagram_approval_state
    
    if not approval_state['section_titles']:
        for i in range(len(doc_generator.pending_mermaid_codes)):
            if len(doc_generator.pending_mermaid_codes) > 1:
                approval_state['section_titles'].append(f"Process {i+1}")
            else:
                approval_state['section_titles'].append("Process Flow Diagram")
    
    if approval_state['current_step'] == 'editing':
        editor = MermaidEditor()
        total_diagrams = len(doc_generator.pending_mermaid_codes)
        st.progress(0, text=f"Review and Edit Diagrams (0/{total_diagrams} completed)")
        
        editor_results = editor.show_batch_editor(
            mermaid_codes=doc_generator.pending_mermaid_codes,
            section_titles=approval_state['section_titles']
        )
        
        all_completed = True
        approved_count = 0
        
        for i, result in enumerate(editor_results):
            if result['approved']:
                approved_count += 1
                if i >= len(approval_state['approved_diagrams']):
                    approval_state['approved_diagrams'].append(result)
                else:
                    approval_state['approved_diagrams'][i] = result
                    
                if i >= len(approval_state['edited_codes']):
                    approval_state['edited_codes'].append(result['mermaid_code'])
                else:
                    approval_state['edited_codes'][i] = result['mermaid_code']
                    
            elif result.get('skip_diagram', False):
                approved_count += 1
                if i >= len(approval_state['approved_diagrams']):
                    approval_state['approved_diagrams'].append({'skipped': True})
                else:
                    approval_state['approved_diagrams'][i] = {'skipped': True}
            else:
                all_completed = False
        
        if approved_count > 0:
            progress = approved_count / total_diagrams
            st.progress(progress, text=f"Review and Edit Diagrams ({approved_count}/{total_diagrams} completed)")
        
        if all_completed:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Continue with Document Generation", type="primary"):
                    approval_state['current_step'] = 'approved'
                    _update_document_generator_diagrams(doc_generator, approval_state)
                    st.rerun()
            
            with col2:
                if st.button("🔄 Start Over"):
                    st.session_state.diagram_approval_state = {
                        'current_step': 'editing',
                        'approved_diagrams': [],
                        'edited_codes': [],
                        'section_titles': []
                    }
                    st.session_state.approved_mermaid_diagrams = []
                    st.rerun()
            
            with col3:
                if st.button("⏭️ Skip All Diagrams"):
                    approval_state['current_step'] = 'skipped'
                    doc_generator.pending_mermaid_codes = []
                    doc_generator.pending_dot_codes = []
                    st.session_state.approved_mermaid_diagrams = []
                    st.rerun()
        
        return False
    
    elif approval_state['current_step'] == 'approved':
        st.success("🎉 Diagrams approved! Generating your document...")
        return True
    
    elif approval_state['current_step'] == 'skipped':
        st.info("⏭️ Diagrams skipped. Generating document without process flow diagrams...")
        return True
    
    return False

def _update_document_generator_diagrams(doc_generator, approval_state):
    """
    Update the document generator with the approved and edited diagrams
    
    Args:
        doc_generator: DocumentGenerator instance
        approval_state: Current approval state with user edits
    """
    try:
        from PIL import Image as PILImage
        
        approved_diagrams = []
        approved_codes = []
        
        # Clear previous approved diagrams in session state
        st.session_state.approved_mermaid_diagrams = []
        
        for i, diagram_data in enumerate(approval_state['approved_diagrams']):
            if diagram_data.get('skipped', False):
                continue
                
            if i < len(approval_state['edited_codes']):
                mermaid_code = approval_state['edited_codes'][i]
                
                # Store approved Mermaid code in session state
                st.session_state.approved_mermaid_diagrams.append({
                    'section_title': approval_state['section_titles'][i],
                    'mermaid_code': mermaid_code
                })
                
                # Convert Mermaid back to DOT for diagram generation
                dot_code = _convert_mermaid_to_dot(mermaid_code)
                
                # Generate the visual diagram
                diagram_image = doc_generator._create_graphviz_diagram(dot_code)
                
                if diagram_image:
                    approved_diagrams.append(diagram_image)
                    approved_codes.append(dot_code)
                    logging.info(f"Generated approved diagram {i+1} from edited Mermaid code")
        
        # Update the document generator with approved diagrams
        doc_generator.approved_diagrams = approved_diagrams
        doc_generator.pending_dot_codes = approved_codes
        
        logging.info(f"Updated document generator with {len(approved_diagrams)} approved diagrams")
        
    except Exception as e:
        logging.error(f"Error updating document generator diagrams: {e}")
        st.error(f"Error processing approved diagrams: {e}")

def _convert_mermaid_to_dot(mermaid_code: str) -> str:
    """
    Convert Mermaid code back to DOT format for diagram generation
    
    Args:
        mermaid_code: Mermaid diagram code
        
    Returns:
        DOT language code
    """
    try:
        import re
        dot_lines = [
            "digraph ProcessFlow {",
            "    rankdir=TB;",
            "    node [shape = box, style=filled, fillcolor=lightblue];",
            "    edge [color=darkblue];"
        ]
        edge_pattern = r'(\w+)\[([^\]]+)\]\s*-->\s*(\w+)\[([^\]]+)\]'
        edges = re.findall(edge_pattern, mermaid_code)
        for from_id, from_label, to_id, to_label in edges:
            dot_lines.append(f'    "{from_label}" -> "{to_label}";')
        
        if not edges:
            simple_pattern = r'(\w+)\s*-->\s*(\w+)'
            simple_edges = re.findall(simple_pattern, mermaid_code)
            for from_node, to_node in simple_edges:
                dot_lines.append(f'    "{from_node}" -> "{to_node}";')
        
        dot_lines.append("}")
        return '\n'.join(dot_lines)
    except Exception as e:
        logging.warning(f"Failed to convert Mermaid to DOT: {e}")
        return """digraph ProcessFlow {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    "Start" -> "End";
}"""

def reset_diagram_approval_state():
    """Reset the diagram approval workflow state"""
    if 'diagram_approval_state' in st.session_state:
        del st.session_state.diagram_approval_state
    if 'approved_mermaid_diagrams' in st.session_state:
        del st.session_state.approved_mermaid_diagrams