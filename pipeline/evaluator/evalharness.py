from lm_eval.models.huggingface import HFLM
import lm_eval
from lm_eval.tasks import TaskManager
from copy import deepcopy
class LMEvalHarness:
    def __init__(self, lm_eval_cfg): 
        self.lm_eval_cfg = deepcopy(lm_eval_cfg)
        self.lm_task_manager = TaskManager()
    def evaluate(self, model):

        hflm = HFLM(pretrained=model.model, tokenizer=model.tokenizer, trust_remote_code=True)
        lm_eval_resutls = lm_eval.simple_evaluate(
            model=hflm, device='cuda', task_manager=self.lm_task_manager, **self.lm_eval_cfg
        )
        return lm_eval_resutls
        
