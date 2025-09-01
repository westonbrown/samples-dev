"""
Model Selector Tool for Intelligent Model Switching

This tool lets the local model analyze user queries to determine task complexity
and select the most appropriate model provider.
"""

from typing import Optional, Dict, Any
from strands import tool, Agent
from strands.models.llamacpp import LlamaCppModel
import logging
import json
from config import BEDROCK_MODEL_ID, LLAMACPP_URL, SESSION_ID

logger = logging.getLogger(__name__)

_local_model_instance = None


@tool
def select_model(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Use local LLM to analyze query complexity and select appropriate model.

    The local model will determine if the query requires:
    - Simple/fast processing (use local model)
    - Complex processing requiring remote capabilities (use remote/cloud model)

    Args:
        query: The user's query to analyze
        context: Optional context about the conversation

    Returns:
        Dictionary containing:
        - provider: Selected provider ("llamacpp" or "bedrock")
        - reasoning: Explanation from the local model
        - requires_remote: Boolean indicating if remote model is needed
        - task_type: Classification of task complexity
    """

    logger.info(f"Using local model to analyze query: {query[:50]}...")

    # Fast path: Check for obvious local model cases
    query_lower = query.lower()
    vehicle_keywords = ["climate", "temperature", "ac", "heat", "window", "seat", "light", "drive mode"]
    if any(keyword in query_lower for keyword in vehicle_keywords):
        return {
            "provider": "llamacpp",
            "model": "default",
            "reasoning": "Vehicle control command - always use local",
            "requires_remote": False,
            "task_type": "simple"
        }

    # System prompt for the local model to analyze complexity
    analysis_prompt = """Analyze query complexity and return JSON:
{"requires_remote_model": bool, "reasoning": "brief explanation", "task_type": "simple|moderate|complex"}

USE LOCAL MODEL (default) for:
- Vehicle controls (climate/windows/seats/lights/modes)
- Calendar/scheduling tasks
- Questions needing <100 word answers
- Tool-handled operations
- Real-time responses needed
- Privacy-sensitive queries

USE REMOTE MODEL only if ANY of these apply:
- Creative writing >200 words requested
- Multi-step analysis with sub-tasks
- Code generation >50 lines
- Academic/research depth explicitly needed
- User says "detailed", "comprehensive", or "in-depth"
- Complex mathematical proofs

Always prefer LOCAL unless remote is clearly needed."""

    try:
        global _local_model_instance
        if _local_model_instance is None:
            _local_model_instance = LlamaCppModel(
                base_url=LLAMACPP_URL,
                model_id="default",
                timeout=120.0,  # Increased timeout for container environment
                params={
                    "temperature": 0.1,  # Lower for faster analysis
                    "max_tokens": 50,  # Optimized for faster response
                    "top_k": 10,  # Faster sampling
                    "response_format": {"type": "json_object"},
                },
            )

        local_model = _local_model_instance

        # Create analysis agent
        analysis_agent = Agent(
            model=local_model, 
            system_prompt=analysis_prompt, 
            callback_handler=None,
            trace_attributes={"session.id", SESSION_ID},
        )

        # Prepare analysis query
        analysis_query = f"Analyze this query: '{query}'"
        if context:
            analysis_query += f"\nContext: {context}"

        # Get analysis from local model
        result = analysis_agent(analysis_query)

        try:
            response_text = str(result).strip()
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_text = response_text[start:end]
                analysis = json.loads(json_text)
            else:
                raise json.JSONDecodeError("No JSON found", response_text, 0)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse JSON from local model: {e}, using keyword-based fallback"
            )
            response_lower = str(result).lower()

            # Default to local model for keyword-based fallback
            requires_remote = False

            analysis = {
                "requires_remote_model": requires_remote,
                "reasoning": f"Keyword-based analysis: {'Complex task detected' if requires_remote else 'Simple task detected'}",
                "task_type": "complex" if requires_remote else "simple",
            }

        # Check if remote model is actually needed
        # NOTE: Currently disabled due to Bedrock credential issues
        # Uncomment the line below to enable remote model selection
        # if analysis.get("requires_remote_model", False):
        if False:  # TODO: Enable when Bedrock credentials are configured
            provider = "bedrock"
            model = BEDROCK_MODEL_ID
        else:
            provider = "llamacpp"
            model = "default"

        return {
            "provider": provider,
            "model": model,
            "reasoning": analysis.get("reasoning", "No reasoning provided"),
            "requires_remote": analysis.get("requires_remote_model", False),
            "task_type": analysis.get("task_type", "unknown"),
        }

    except Exception as e:
        logger.error(f"Error in model selection: {e}")
        # Fallback to local model on error
        return {
            "provider": "llamacpp",
            "model": "default",
            "reasoning": f"Error during analysis: {str(e)}. Defaulting to local model.",
            "requires_remote": False,
            "task_type": "unknown",
        }
