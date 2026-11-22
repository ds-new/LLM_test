import torch, inspect
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
print(inspect.signature(OptimizerParamScheduler.__init__))
def run(init_lr, warm):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-6, weight_decay=0.1)
    kw = dict(init_lr=init_lr, max_lr=1e-6, min_lr=0.0, lr_warmup_steps=warm, lr_decay_steps=20, lr_decay_style='constant',
              start_wd=0.1, end_wd=0.1, wd_incr_steps=20, wd_incr_style='constant')
    sig = inspect.signature(OptimizerParamScheduler.__init__).parameters
    kw = {k: v for k, v in kw.items() if k in sig}
    sch = OptimizerParamScheduler(opt, **kw)
    out = []
    for i in range(5):
        out.append(opt.param_groups[0]['lr'])   # what verl logs / uses for this step
        sch.step(1)
    return out
print('old (init 0, warmup 3):   ', ['%.3g' % x for x in run(0.0, 3)])
print('new (init lr/3, warmup 1):', ['%.3g' % x for x in run(3.33333e-07, 1)])
