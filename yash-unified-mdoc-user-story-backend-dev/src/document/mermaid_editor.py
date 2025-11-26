import streamlit as st
import datetime
import time
import re
from typing import Dict, Any

class MermaidEditor:
    """
    Interactive Mermaid diagram editor with live preview and approval workflow
    """
    
    def __init__(self):
        """Initialize the Mermaid editor"""
        self.default_template = """graph TD
    A[Start] --> B{Decision Point}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E[End]"""
        
        # Initialize session state for approved diagrams
        if 'approved_mermaid_diagrams' not in st.session_state:
            st.session_state.approved_mermaid_diagrams = []
    
    def _clean_mermaid_code(self, code: str) -> str:
        """Clean and validate Mermaid code with comprehensive syntax fixing"""
        if not code:
            return self.default_template
            
        # Remove any remaining markdown code blocks
        code = code.strip()
        if code.startswith("```mermaid"):
            code = code[10:].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
            
        # Remove any extra whitespace and empty lines
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        code = '\n'.join(lines)
        
        # Basic validation - ensure it's not empty
        if not code.strip():
            return self.default_template
        
        # Fix common syntax issues
        code = self._fix_mermaid_syntax(code)
            
        return code.strip()
    
    def _fix_mermaid_syntax(self, code: str) -> str:
        """Fix common Mermaid syntax issues"""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Fix node IDs that might have invalid characters
            # Replace problematic characters in node IDs
            line = re.sub(r'([A-Z]\w*)\s*\[([^\]]*)\]', lambda m: f"{self._clean_node_id(m.group(1))}[{self._clean_label(m.group(2))}]", line)
            
            # Fix arrow syntax
            line = re.sub(r'\s*-->\s*', ' --> ', line)
            line = re.sub(r'\s*---\s*', ' --- ', line)
            
            # Fix decision node syntax
            line = re.sub(r'([A-Z]\w*)\s*\{([^\}]*)\}', lambda m: f"{self._clean_node_id(m.group(1))}{{{self._clean_label(m.group(2))}}}", line)
            
            # Fix label syntax on arrows
            line = re.sub(r'\|\s*([^|]*)\s*\|', lambda m: f"|{self._clean_label(m.group(1))}|", line)
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _clean_node_id(self, node_id: str) -> str:
        """Clean node ID to ensure it's valid"""
        # Remove invalid characters and ensure it starts with a letter
        cleaned = re.sub(r'[^A-Za-z0-9_]', '', node_id)
        if not cleaned or not cleaned[0].isalpha():
            cleaned = 'N' + cleaned
        return cleaned[:20]  # Limit length
    
    def _clean_label(self, label: str) -> str:
        """Clean node label to remove problematic characters"""
        # Remove or escape problematic characters
        label = label.replace('"', "'").replace('\n', ' ').replace('\r', '')
        # Remove excess whitespace
        label = ' '.join(label.split())
        return label[:50]  # Limit length
    
    def _validate_mermaid_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Mermaid syntax and return (is_valid, error_message)"""
        lines = code.split('\n')
        
        # Check if it starts with a valid diagram type
        first_line = lines[0].strip() if lines else ""
        valid_starts = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'journey', 'pie']
        
        if not any(first_line.startswith(start) for start in valid_starts):
            return False, f"Invalid diagram type. Must start with one of: {', '.join(valid_starts)}"
        
        # Check for basic syntax issues
        for i, line in enumerate(lines[1:], 2):  # Skip first line
            line = line.strip()
            if not line:
                continue
                
            # Check for unmatched brackets
            if line.count('[') != line.count(']'):
                return False, f"Unmatched square brackets on line {i}: {line}"
            if line.count('{') != line.count('}'):
                return False, f"Unmatched curly brackets on line {i}: {line}"
            if line.count('(') != line.count(')'):
                return False, f"Unmatched parentheses on line {i}: {line}"
        
        return True, ""
    
    def _analyze_diagram(self, mermaid_code: str) -> Dict[str, int]:
        """Analyze diagram complexity"""
        try:
            lines = mermaid_code.split('\n')
            nodes = set()
            connections = 0
            
            for line in lines:
                line = line.strip()
                if '-->' in line or '---' in line:
                    connections += 1
                    # Extract node identifiers (basic parsing)
                    parts = line.split('-->')
                    if len(parts) == 2:
                        left = parts[0].strip().split('[')[0].split('(')[0].split('{')[0].split('|')[0]
                        right = parts[1].strip().split('[')[0].split('(')[0].split('{')[0].split('|')[0]
                        nodes.add(left)
                        nodes.add(right)
            
            return {'nodes': len(nodes), 'connections': connections}
        except:
            return {'nodes': 0, 'connections': 0}

    def show_editor_interface(self, initial_mermaid_code: str = None, 
                            section_title: str = "Process Flow Diagram",
                            editor_key: str = "mermaid_editor") -> Dict[str, Any]:
        
        # Clean the initial code
        cleaned_initial_code = self._clean_mermaid_code(initial_mermaid_code) if initial_mermaid_code else self.default_template
        
        # Validate the cleaned code
        is_valid, error_msg = self._validate_mermaid_syntax(cleaned_initial_code)
        
        if not is_valid:
            st.error(f"❌ Syntax Error: {error_msg}")
            st.warning("🔧 Using default template instead")
            cleaned_initial_code = self.default_template

        # Initialize session state
        editor_state_key = f"{editor_key}_state"
        if editor_state_key not in st.session_state:
            st.session_state[editor_state_key] = {
                'mermaid_code': cleaned_initial_code,
                'last_update': cleaned_initial_code,
                'approved': False,
                'show_editor': False,
                'diagram_count': 1
            }

        editor_state = st.session_state[editor_state_key]

        # Show diagram preview
        current_code = editor_state.get('last_update', editor_state['mermaid_code'])
        current_code = self._clean_mermaid_code(current_code)
        
        # Validate current code
        is_valid, error_msg = self._validate_mermaid_syntax(current_code)

        if current_code.strip() and is_valid:
            try:
                import streamlit_mermaid as stmd
                
                # Create a unique key for each render to avoid caching issues
                refresh_key = f"{editor_key}_preview_{abs(hash(current_code))}"
                time.sleep(0.1)
                # Render the Mermaid diagram
                stmd.st_mermaid(current_code, height=500, key=refresh_key)

                # Show diagram statistics
                stats = self._analyze_diagram(current_code)
                st.caption(f"📈 Nodes: {stats['nodes']}, Connections: {stats['connections']}")
                
            except ImportError:
                st.error("❌ streamlit-mermaid not installed. Run: `pip install streamlit-mermaid`")
                
            except Exception as e:
                st.error(f"❌ Rendering Error: {str(e)}")
                st.error("**Raw code being rendered:**")
                st.code(current_code, language="text")
                
                # Show the problematic lines
                lines = current_code.split('\n')
                st.error("**Code breakdown:**")
                for i, line in enumerate(lines, 1):
                    st.write(f"Line {i}: `{line}`")
                
        elif not is_valid:
            st.error(f"❌ Invalid Mermaid Syntax: {error_msg}")
            st.error("**Problematic code:**")
            st.code(current_code, language="text")
        else:
            st.info("📝 No Mermaid code available to preview.")

        # Advanced editor in expander
        with st.expander("🔧 Advanced Editor:"):
            # Show original vs cleaned code
            # if initial_mermaid_code:
            #     st.markdown("**Original extracted code:**")
            #     st.code(initial_mermaid_code[:500] + "..." if len(initial_mermaid_code) > 500 else initial_mermaid_code, language="text")
                
            #     st.markdown("**Cleaned code:**")
            #     st.code(current_code, language="text")
            
            mermaid_code = st.text_area(
                "Edit Mermaid Diagram Code:",
                value=editor_state['mermaid_code'],
                height=300,
                key=f"{editor_key}_code_area",
                help="Edit your diagram code here. Syntax will be validated on update."
            )
            
            # Update button
            if st.button("🔄 Update Preview", key=f"{editor_key}_update"):
                cleaned_code = self._clean_mermaid_code(mermaid_code)
                is_valid, error_msg = self._validate_mermaid_syntax(cleaned_code)
                
                if is_valid:
                    editor_state['mermaid_code'] = cleaned_code
                    editor_state['last_update'] = cleaned_code
                    st.success("✅ Preview updated!")
                    st.rerun()
                else:
                    st.error(f"❌ Cannot update: {error_msg}")

            # External editor link
            st.markdown("**External Editor:**")
            st.markdown("Edit your diagram using the code at: [Mermaid Live Editor](https://mermaid.live/)")
        
        # Return current state
        return {
            'approved': editor_state.get('approved', False),
            'mermaid_code': current_code if is_valid else self.default_template,
            'skip_diagram': False
        }