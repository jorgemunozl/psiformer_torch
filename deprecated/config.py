from pydantic import BaseModel, Field


class Config(BaseModel):
    n_layer: int = Field(default=4,
                         description="")
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768


class outPut(BaseModel):
    max_length: int = Field(default=4,
                            description="")
    max_return_seq: int = Field(default=4,
                                description="")
