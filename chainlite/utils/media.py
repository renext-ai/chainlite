"""Media and prompt construction helpers for ChainLite."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from typing import Any, Union

import jinja2
from loguru import logger
from pydantic_ai import BinaryContent, ImageUrl


async def process_media_item(item: Any) -> Union[ImageUrl, BinaryContent, str]:
    """Process one media input into a Pydantic AI content object."""
    if isinstance(item, (ImageUrl, BinaryContent)):
        return item

    if isinstance(item, str):
        if item.startswith("data:"):
            try:
                header, encoded = item.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
                data = base64.b64decode(encoded)
                return BinaryContent(data=data, media_type=media_type)
            except Exception as e:
                logger.warning(f"Failed to process base64 string: {e}")
                return item

        if item.startswith(("http://", "https://")):
            return ImageUrl(item)

        if os.path.exists(item):
            mime_type, _ = mimetypes.guess_type(item)
            if not mime_type:
                mime_type = "image/jpeg"
            try:

                def _read() -> bytes:
                    with open(item, "rb") as f:
                        return f.read()

                file_data = await asyncio.to_thread(_read)
                return BinaryContent(data=file_data, media_type=mime_type)
            except Exception as e:
                logger.warning(f"Failed to read local file {item}: {e}")
                raise e

        return ImageUrl(item)

    return str(item)


async def build_prompt(
    prompt_template: str,
    input_data: dict,
) -> Union[str, list[Union[str, BinaryContent, ImageUrl]]]:
    """Render text prompt and append optional image/media parts."""
    template = jinja2.Template(prompt_template)
    prompt_str = template.render(**input_data)

    content_list: list[Union[str, BinaryContent, ImageUrl]] = [prompt_str]

    images = input_data.get("images")
    if images and isinstance(images, list):
        for img in images:
            content_list.append(await process_media_item(img))

    image_url = input_data.get("image_url")
    if image_url:
        content_list.append(await process_media_item(image_url))

    if len(content_list) > 1:
        return content_list

    return prompt_str
