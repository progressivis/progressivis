from .utils import wait_for_change, update_widget
from .data_table import DataTable
from progressivis.core import JSONEncoderNp as JSON
from progressivis.table.paging_helper import PagingHelper


debug_console = None
_dmp = JSON.dumps
# https://datatables.net/examples/basic_init/alt_pagination.html


async def pagination(dt, tbl):
    while True:
        await wait_for_change(dt, 'page')
        info = dt.page
        _len = len(tbl)
        helper = PagingHelper(tbl)
        data = helper.get_page(info['start'], info['end'])
        # data = tbl.iloc[info['start']:info['end']].to_json(orient='datatable')
        js_data = {'data': data, 'recordsTotal': _len, 'recordsFiltered': _len,
                   'length': _len, 'draw': info['draw'], 'page': info['page']}
        dt.data = _dmp(js_data)


class SlotWg(DataTable):
    def __init__(self, module, slot_name, dconsole=None):
        global debug_console
        debug_console = dconsole
        self.module = module
        self.slot_name = slot_name
        super().__init__()

    async def refresh(self):
        tbl = self.module.get_data(self.slot_name)
        if tbl is None:
            return
        if not self.columns:
            # self.dt_id = "dt_"+self.module.name+"_"+self.slot_name
            await update_widget(self, 'dt_id',
                                "dt_"+self.module.name+"_"+self.slot_name)
            if isinstance(tbl, dict):
                # self.columns = _dmp(['index']+list(tbl.keys()))
                await update_widget(self, 'columns',
                                    _dmp(['index']+list(tbl.keys())))
            else:  # table
                # self.columns = _dmp(['index']+tbl.columns)
                await update_widget(self, 'columns',
                                    _dmp(['index']+tbl.columns))
                # aio.create_task(pagination(self, tbl))
        if isinstance(tbl, dict):
            # self.data = _dmp({'draw':1,'recordsTotal': 1,
            #                   'recordsFiltered': 1,
            #             'data': [[0]+list(tbl.values())]})
            await update_widget(self, 'data',
                                _dmp({'draw': 1,
                                      'recordsTotal': 1,
                                      'recordsFiltered': 1,
                                      'data': [[0]+list(tbl.values())]}))
        else:  # table
            _len = len(tbl)
            size = 10
            info = dict(start=0, end=size, draw=1)
            if self.page:
                info.update(self.page)
            start = info['start']
            end = info['end']
            draw = info['draw']
            helper = PagingHelper(tbl)
            data = helper.get_page(start, end)
            # data = tbl.iloc[start:end].to_json(orient='datatable')
            # self.data = _dmp({'data': data, 'recordsTotal': _len,
            #                  'recordsFiltered': _len, 'length': _len,
            #                  'draw':draw})
            await update_widget(self, 'data',
                                _dmp({'data': data,
                                      'recordsTotal': _len,
                                      'recordsFiltered': _len,
                                      'length': _len,
                                      'draw': draw}))
