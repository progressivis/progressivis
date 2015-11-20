from progressivis import Constant, ProgressiveError

class Variable(Constant):
    def __init__(self, df, **kwds):
        if len(df)==0:
            raise ProgressiveError('Initial dataframe should not be empty')
        super(Variable, self).__init__(df, **kwds)

    def add_input(self, input):
        if not isinstance(input,dict):
            raise ProgressiveError('Expecting a dictionary')
        last = self.last_row(self._df).to_dict()
        error = ''
        for (k, v) in input.iteritems():
            if k in last:
                last[k] = v
            else:
                error += 'Invalid key %s ignored. '%k
        with self.lock:
            run_number = self.scheduler().run_number()+1
            last[self.UPDATE_COLUMN] = run_number
            self._df[self._df.index[-1]+1] = last
        return error
    
