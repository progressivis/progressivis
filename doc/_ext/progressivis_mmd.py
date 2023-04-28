# type: ignore

import sphinx
from sphinxcontrib.mermaid import Mermaid
from sphinx.ext.graphviz import Graphviz
from sphinx.locale import __
from sphinx.util.i18n import search_image_for_language

from docutils.nodes import Node


class ProgressivisMMD(Mermaid):
    def get_mm_code(self):
        py_code = super().get_mm_code()
        locals_ = dict(scheduler=None)
        exec(py_code, globals(), locals_)
        return locals_["scheduler"].to_mermaid()


class ProgressivisDOT(Graphviz):
    def run(self) -> list[Node]:
        if self.arguments:
            document = self.state.document
            if self.content:
                return [document.reporter.warning(
                    __('Graphviz directive cannot have both content and '
                       'a filename argument'), line=self.lineno)]
            argument = search_image_for_language(self.arguments[0], self.env)
            rel_filename, filename = self.env.relfn2path(argument)
            self.env.note_dependency(rel_filename)
            try:
                with open(filename, encoding='utf-8') as fp:
                    py_code = fp.read()
            except OSError:
                return [document.reporter.warning(
                    __('External Graphviz file %r not found or reading '
                       'it failed') % filename, line=self.lineno)]
        else:
            py_code = self.content
            rel_filename = None
            if not py_code.strip():
                return [self.state_machine.reporter.warning(
                    __('Ignoring "progressiv_dot" directive without content.'),
                    line=self.lineno)]
        self.arguments = None
        locals_ = dict(scheduler=None)
        exec(py_code, globals(), locals_)
        dotcode = locals_["scheduler"].to_graphviz()
        self.content = [dotcode]
        return super().run()


def setup(app):
    app.add_directive("progressivis_mmd", ProgressivisMMD)
    app.add_directive("progressivis_dot", ProgressivisDOT)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
