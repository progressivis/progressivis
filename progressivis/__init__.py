from progressivis.core import *

import logging

__all__ = ["log_level", "ProgressiveError","Scheduler", "MTScheduler", 
           "Slot", "SlotDescriptor", "Module", "connect", "StorageManager",
           "DataFrameModule", "Constant", "Print", "Wait", "Merge", "NIL" ]


# Magic spell to avoid the message 'No handlers could be found for logger X.Y.Z'
logging.getLogger('progressivis').addHandler(logging.NullHandler())

def log_level(level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('progressivis').addHandler(ch)
    logging.getLogger('progressivis').setLevel(level)

