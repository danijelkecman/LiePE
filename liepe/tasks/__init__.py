from . import copy_delay, induction, modular, timeseries

TASKS = {
    "copy_delay": copy_delay.make_batch,
    "induction": induction.make_batch,
    "modular": modular.make_batch,
    "timeseries": timeseries.make_batch,
}
