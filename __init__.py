from comfy_api.latest import ComfyExtension, io

from .h3_chain_node import MiniMaxH3ChainVideo


class MiniMaxChainExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MiniMaxH3ChainVideo]


async def comfy_entrypoint() -> MiniMaxChainExtension:
    return MiniMaxChainExtension()
