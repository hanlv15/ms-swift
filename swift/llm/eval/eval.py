from typing import Any, Dict, List, Union

from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager

from ..argument import EvalArguments
from ..base import SwiftPipeline


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args

    def run_opencompass(self):
        pass

    def run_vlmeval(self):
        pass


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
