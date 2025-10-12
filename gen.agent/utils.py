from re import L
import filetype
from pydub import AudioSegment
import io
import json
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmRequest, LlmResponse
from typing import Optional, Type, List
from google.genai.types import Content, Part

from pydantic import BaseModel, Field, create_model

def fill_json_schema(schema, data):
    def fill_recursive(schema_part, data_part):
        if isinstance(schema_part, dict):
            result = {}
            for key, value in schema_part.items():
                if key in data_part:
                    result[key] = fill_recursive(value, data_part[key])
                else:
                    if isinstance(value, dict):
                        result[key] = fill_recursive(value, {})
                    elif isinstance(value, list):
                        result[key] = []
                    else:
                        result[key] = None
            return result
        elif isinstance(schema_part, list):
            if len(schema_part) == 0:
                # Empty schema list - return data as-is if it's a list
                return data_part if isinstance(data_part, list) else []
            elif isinstance(data_part, list):
                # Schema has template, data is a list - process each item
                return [fill_recursive(schema_part[0], item) for item in data_part]
            else:
                # Schema expects list but data isn't a list
                return []
        else:
            # If the data_part is a string, replace \" with '
            if isinstance(data_part, str):
                return data_part.replace('\\"', "'")
            return data_part if data_part is not None else schema_part

    return fill_recursive(schema, data)

def convert_to_wav(data):

    # Load the audio file

    audio = AudioSegment.from_file(io.BytesIO(data))

    # Use the frame rate, sample width, and channels from the input audio

    wav_audio = (
        audio.set_frame_rate(audio.frame_rate).set_sample_width(2).set_channels(1)
    )

    # Export the WAV file to an in-memory file-like object

    wav_io = io.BytesIO()

    wav_audio.export(wav_io, format="wav")

    # Return the in-memory file-like object

    return wav_io.getvalue()

def get_mime_type(file_bytes):

    kind = filetype.guess(file_bytes)
    if kind:
        return kind.mime
    else:
        return "application/octet-stream"

def format_response(response, data_format, lower_case=False):

    response = (
        response.candidates[0]
        .content.parts[0]
        .text.replace("`", "")
        .replace("json", "")
        .replace("sql", "")
        .replace("'", '\\"')
        .replace("\\n", " ")
        .strip()
    )

    if lower_case:
        response = response.lower()

    if data_format is None:
       return json.loads(response)
    
    if isinstance(data_format, str):
        data_format = json.loads(data_format)

    return fill_json_schema(data_format, json.loads(response)) 

def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM response after it's received."""

    print("LLM CALL COMPLETED: [" + callback_context.agent_name + "]")

    # --- Inspection ---
    if llm_response.content and llm_response.content.parts:
        # Iterate through all parts, not just the first one
        original_text = ""
        function_call_name = ""
        for part in llm_response.content.parts:
            if part.text:
                original_text = original_text + part.text
            if part.function_call:
                function_call_name = part.function_call.name

        if function_call_name:
            appendHistory(callback_context, "(FUNC): " + function_call_name)
        if original_text:
            appendHistory(callback_context, "(RES): " + original_text)

    return None

def before_model_callback(
    callback_context: CallbackContext,llm_request: LlmRequest
) -> Optional[None]:

    """Inspects/modifies the LLM request or skips the call."""

    print("LLM CALL STARTED: [" + callback_context.agent_name + "]")

    # Inspect the last user message in the request contents
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:

            original_text = ""
            for part in llm_request.contents[-1].parts:
                if part.text:
                    original_text = original_text + part.text

            chat_history = callback_context.state.get("chat_history", [])

            if original_text:
                if not chat_history or len(chat_history) == 0:
                    appendHistory(callback_context, "(USER): " + str(original_text))
                else:
                    appendHistory(callback_context, "(REQ): " + str(original_text))
    return None

def appendHistory(callback_context: CallbackContext, text: str):
    agent_name = callback_context.agent_name.upper()
    callback_context.state.update({
        "chat_history": callback_context.state.get("chat_history", []) + ["[" + agent_name + "] " + text]
    })
    return None

def json_to_adk_schema(json_schema, model_name: str = "DynamicOutputModel") -> Type[BaseModel]:
    # Handle None or empty input
    if not json_schema:
        return None
    
    # If it's a string, try to parse it as JSON
    if isinstance(json_schema, str):
        try:
            json_schema = json.loads(json_schema)
        except json.JSONDecodeError:
            # If it fails, return None
            return None
    
    # If it's not a dict after parsing, return None
    if not isinstance(json_schema, dict):
        return None
    
    def get_python_type(field_schema):
        """Convert field schema to Python type annotation."""
        
        # Handle list type (e.g., [], [""], [{}], [0])
        if isinstance(field_schema, list):
            if len(field_schema) == 0:
                # Empty list: default to List[str]
                return List[str]
            elif len(field_schema) == 1:
                # Determine item type from the first element
                item_schema = field_schema[0]
                if isinstance(item_schema, str):
                    return List[str]
                elif isinstance(item_schema, int):
                    return List[int]
                elif isinstance(item_schema, float):
                    return List[float]
                elif isinstance(item_schema, bool):
                    return List[bool]
                elif isinstance(item_schema, dict):
                    if item_schema:  # Non-empty dict
                        nested_model = json_to_adk_schema(item_schema, f"{model_name}Item")
                        return List[nested_model]
                    else:  # Empty dict
                        return List[dict]
                else:
                    return List[str]
            else:
                # Multiple elements: use first element type
                return get_python_type([field_schema[0]])
        
        # Handle simple string type (e.g., "string", "integer")
        if isinstance(field_schema, str):
            type_map = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            return type_map.get(field_schema, str)
        
        # Handle dict with type field
        if not isinstance(field_schema, dict):
            return str
        
        field_type = field_schema.get("type", "string")
        
        # Type mapping
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool
        }
        
        # Handle array type
        if field_type == "array":
            items_schema = field_schema.get("items", {})
            item_type = get_python_type(items_schema)
            return List[item_type]
        
        # Handle object type (nested model)
        if field_type == "object" and "properties" in field_schema:
            nested_props = field_schema["properties"]
            nested_model = json_to_adk_schema(nested_props, f"{model_name}Nested")
            return nested_model
        
        # Handle primitive types
        return type_map.get(field_type, str)
    
    # Build field definitions for create_model
    field_definitions = {}
    
    for field_name, field_schema in json_schema.items():
        # Handle simple string type
        if isinstance(field_schema, str):
            python_type = get_python_type(field_schema)
            field_definitions[field_name] = (Optional[python_type], Field(default=None, description=""))
        # Handle list type
        elif isinstance(field_schema, list):
            python_type = get_python_type(field_schema)
            field_definitions[field_name] = (Optional[python_type], Field(default=None, description=""))
        else:
            # Handle dict with full schema
            python_type = get_python_type(field_schema)
            is_required = field_schema.get("required", False) if isinstance(field_schema, dict) else False
            description = field_schema.get("description", "") if isinstance(field_schema, dict) else ""
            
            if is_required:
                field_definitions[field_name] = (python_type, Field(description=description))
            else:
                field_definitions[field_name] = (Optional[python_type], Field(default=None, description=description))
    
    # Create and return the Pydantic model
    return create_model(model_name, **field_definitions)
