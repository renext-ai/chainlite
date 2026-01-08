from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ChainLiteConfig(BaseModel):
    """
    Configuration model for ChainLite.
    """

    config_name: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    llm_model_name: str
    temperature: Optional[float] = None
    output_parser: Optional[List[Dict[str, Any]]] = None
    use_history: Optional[bool] = False
    session_id: str = "unused"
    redis_url: Optional[str] = None
    max_retries: Optional[int] = 1
