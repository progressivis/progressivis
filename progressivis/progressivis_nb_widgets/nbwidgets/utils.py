import asyncio as aio
import ipywidgets as widgets
from progressivis.core import asynchronize
# cf. https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Asynchronous.html
def wait_for_change(widget, value):
    future = aio.Future()
    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)
    widget.observe(getvalue, value)
    return future

def wait_for_click(btn, cb):
    future = aio.Future()
    def proc_(_):
        future.set_result(True)
        btn.on_click(proc_, remove=True)
        cb()
    btn.on_click(proc_)
    return future


async def update_widget(wg, attr, val):
    await asynchronize(setattr, wg, attr, val)

