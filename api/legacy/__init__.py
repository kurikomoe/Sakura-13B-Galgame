from typing import *
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from transformers import GenerationConfig

from utils import model as M


import logging 

logger = logging.getLogger(__name__)

FloatOrInt = Union[float, int]

class Generate(BaseModel):
    prompt: str
    auto_max_new_tokens: bool = False
    max_tokens_second: int = 0
    # Generation params. If 'preset' is set to different than 'None' the values
    # in presets/preset-name.yaml are used instead of the individual numbers.
    preset: str | None = None
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float | int = 0.1
    top_p: float | int = 0.3
    repetition_penalty: float | int = 1.0
    num_beams: float | int = 1
    typical_p: float | int = 1
    epsilon_cutoff: float | int = 0  # In units of 1e-4
    eta_cutoff: float | int = 0  # In units of 1e-4
    tfs: float | int = 1
    top_a: float | int  = 0
    presence_penalty: float | int = 0
    frequency_penalty: float | int = 0
    repetition_penalty_range: float | int = 0
    top_k: float | int = 40
    min_length: float | int = 0
    no_repeat_ngram_size: float | int = 0
    penalty_alpha: float | int = 0
    length_penalty: float | int = 1
    early_stopping: bool = False
    mirostat_mode: float | int = 0
    mirostat_tau: float | int = 5
    mirostat_eta: float | int = 0.1
    grammar_string: str = ""
    guidance_scale: float | int = 1
    negative_prompt: str = ""
    seed: float | int = -1
    add_bos_token: bool = True
    truncation_length: float | int = 2048
    ban_eos_token: bool = False
    custom_token_bans: str = ""
    skip_special_tokens: bool = True
    stopping_strings: List[Any] = []


router = APIRouter(
    prefix="/api/v1",
)

@router.post("/generate")
async def completions(data: Generate):
    print(data)
    generation_config = GenerationConfig(**data.__dict__)
    logger.info(f"translate: {data.prompt}")
    output = await M.get_model().completion(data.prompt, generation_config)
    logger.info(f"answer: {output}")
    ret = {
        "results": [
            {
                "text": output,
            }
        ],
    }

    json_compatible_item_data = jsonable_encoder(ret)
    return JSONResponse(content=json_compatible_item_data)