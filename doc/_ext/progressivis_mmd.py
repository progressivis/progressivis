# type: ignore

from sphinxcontrib.mermaid import Mermaid


class ProgressivisMMD(Mermaid):
    def get_mm_code(self):
        py_code = super().get_mm_code()
        locals_ = dict(scheduler=None)
        exec(py_code, globals(), locals_)
        return locals_["scheduler"].to_mermaid()


def setup(app):
    app.add_directive("progressivis_mmd", ProgressivisMMD)
