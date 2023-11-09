from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from argparse import ArgumentParser
from transformers import GenerationConfig
from sampler_hijack import hijack_samplers

from pprint import pprint

import logging
from fastapi import FastAPI

from utils import *
from utils import model as M

from fastapi.security import HTTPBasic, HTTPBasicCredentials


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

from api.legacy import router as legacy_router
app.include_router(legacy_router)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="SakuraLLM/Sakura-13B-LNovel-v0.8", help="model huggingface id or local path.")
    parser.add_argument("--use_gptq_model", action="store_true", help="whether your model is gptq quantized.")
    parser.add_argument("--model_version", type=str, default="0.8", help="model version written on huggingface readme, now we have ['0.1', '0.4', '0.5', '0.7', '0.8']")
    parser.add_argument("--data_path", type=str, default="data.txt", help="file path of the text you want to translate.")
    parser.add_argument("--output_path", type=str, default="data_translated.txt", help="save path of the text model translated.")
    parser.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")
    parser.add_argument("--compare_text", action="store_true", help="whether to output with both source text and translated text in order to compare.")
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code.")
    parser.add_argument("--llama", action="store_true", help="whether your model is llama family.")
    args = parser.parse_args()

    # copy k,v from args to cfg
    cfg = M.SakuraModelConfig()
    for k, v in args.__dict__.items():
        cfg.__dict__[k] = v

    pprint(f"Current config: {cfg.__dict__}")

    M.init_model(cfg)

    uvicorn.run(app, host="0.0.0.0", port=5001)




