#!/usr/bin/env python3
"""
Convert existing train.jsonl and test.jsonl to HuggingFace SFTTrainer format.

This script parses the text-formatted training data and converts it to 
the proper HuggingFace format for tool calling with separate messages and tools.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_tools_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract tool definitions from text and convert to HF format."""
    tools = []
    
    # Look for tools in the <tools> section
    tools_match = re.search(r'<tools>(.*?)</tools>', text, re.DOTALL)
    if tools_match:
        tools_content = tools_match.group(1)
        
        # Parse JSON tool definitions
        try:
            # Find JSON objects in the tools content
            json_matches = re.findall(r'\{[^}]*"name"[^}]*\}', tools_content)
            for json_str in json_matches:
                tool_def = json.loads(json_str)
                # Convert to HF format
                hf_tool = {
                    "type": "function",
                    "function": {
                        "name": tool_def["name"],
                        "description": tool_def["description"],
                        "parameters": tool_def["inputSchema"]["json"]
                    }
                }
                tools.append(hf_tool)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse tool definition: {e}")
    
    return tools

def parse_text_to_hf_format(text: str) -> Dict[str, Any]:
    """Parse text format into HuggingFace tool calling format.
    
    Args:
        text: The formatted text with <|im_start|> and <|im_end|> markers
        
    Returns:
        Dict with 'messages' and 'tools' keys for SFTTrainer
    """
    messages = []
    tools = extract_tools_from_text(text)
    
    # Split by message boundaries
    parts = re.split(r'<\|im_start\|>', text)
    
    for i, part in enumerate(parts):
        if not part.strip():
            continue
            
        # Remove the end marker
        part = part.replace('<|im_end|>', '').strip()
        
        # Extract role and content
        role = None
        content = ""
        
        if part.startswith('system\n'):
            role = 'system'
            content = part[7:].strip()  # Remove 'system\n'
            # Simplify system message since tools are separate
            content = "You are an AI assistant with access to various tools. Use them to help users effectively."
            
        elif part.startswith('user\n'):
            role = 'user'
            content = part[5:].strip()  # Remove 'user\n'
            
            # Check if this is a tool result
            if content.startswith('[Tool Result:'):
                # This is a tool result, convert to tool role
                role = 'tool'
                # Try to get tool name from previous message
                tool_name = "unknown_tool"
                if messages and messages[-1].get('role') == 'assistant' and 'tool_calls' in messages[-1]:
                    tool_name = messages[-1]['tool_calls'][0]['function']['name']
                
                # Clean up the content
                content = content.replace('[Tool Result: ', '').replace(']', '').strip()
                
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": content
                })
                continue
                
        elif part.startswith('assistant\n'):
            role = 'assistant'
            content = part[10:].strip()  # Remove 'assistant\n'
            
            # Remove <think> tags if they're empty
            content = re.sub(r'<think>\s*</think>\s*', '', content).strip()
            
            # Check if this contains tool calls (handle multi-line JSON)
            tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            tool_calls_found = re.findall(tool_call_pattern, content, re.DOTALL | re.MULTILINE)
            
            if tool_calls_found:
                # This is a tool call message
                tool_calls = []
                for tool_call_json in tool_calls_found:
                    try:
                        tool_call = json.loads(tool_call_json)
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"]  # Keep as dict
                            }
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing tool call: {e}")
                        continue
                
                if tool_calls:
                    messages.append({
                        "role": "assistant",
                        "tool_calls": tool_calls
                    })
                    continue
            
            # Regular assistant message (remove tool call XML if present)
            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
            
        if role and content:
            messages.append({
                "role": role,
                "content": content
            })
    
    return {
        "messages": messages,
        "tools": tools
    }

def convert_jsonl_to_hf(input_path: str, output_path: str):
    """Convert a JSONL file from text format to HuggingFace SFTTrainer format.
    
    Args:
        input_path: Path to input JSONL file with text format
        output_path: Path to output JSONL file with HF format
    """
    print(f"Converting {input_path} to {output_path}...")
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: {input_path} does not exist")
        return
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                if not text:
                    skipped_count += 1
                    continue
                
                # Parse the text into HF format
                hf_data = parse_text_to_hf_format(text)
                
                if hf_data.get('messages'):
                    outfile.write(json.dumps(hf_data) + '\n')
                    converted_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1
                continue
    
    print(f"✅ Converted {converted_count} examples")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} examples")

def validate_hf_format(file_path: str, max_examples: int = 3):
    """Validate and display examples from the HF format file."""
    print(f"\nValidating {file_path}...")
    
    file = Path(file_path)
    if not file.exists():
        print(f"Error: {file_path} does not exist")
        return
    
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
                
            data = json.loads(line)
            messages = data.get('messages', [])
            tools = data.get('tools', [])
            
            print(f"\nExample {i+1}:")
            print(f"  Messages: {len(messages)}")
            print(f"  Tools: {len(tools)}")
            
            # Show tool names
            if tools:
                tool_names = [tool['function']['name'] for tool in tools]
                print(f"  Tool names: {', '.join(tool_names)}")
            
            # Show message roles
            roles = [msg.get('role') for msg in messages]
            print(f"  Message roles: {' -> '.join(roles)}")
            
            # Check for tool calls
            tool_calls = [msg for msg in messages if 'tool_calls' in msg]
            if tool_calls:
                print(f"  Tool calls found: {len(tool_calls)}")

def main():
    """Main conversion function."""
    
    print("=" * 60)
    print("Converting to HuggingFace SFTTrainer format")
    print("=" * 60)
    
    data_dir = Path("/home/ubuntu/samples-dev/02-samples/14-agentic-ai-at-the-edge/src/data_science_pipeline/data")
    
    # Convert train.jsonl
    train_input = data_dir / "train.jsonl"
    train_output = data_dir / "train_sft.jsonl"
    
    if train_input.exists():
        convert_jsonl_to_hf(str(train_input), str(train_output))
        validate_hf_format(str(train_output), max_examples=2)
    else:
        print(f"Warning: {train_input} not found")
    
    print("\n" + "-" * 40 + "\n")
    
    # Convert test.jsonl
    test_input = data_dir / "test.jsonl"
    test_output = data_dir / "test_sft.jsonl"
    
    if test_input.exists():
        convert_jsonl_to_hf(str(test_input), str(test_output))
        validate_hf_format(str(test_output), max_examples=2)
    else:
        print(f"Warning: {test_input} not found")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("\nGenerated files:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print("\nThese files are ready for use with HuggingFace SFTTrainer.")
    print("They include proper tool_calls format and separate tools definitions.")

if __name__ == "__main__":
    main()