
import matplotlib.pyplot as plt

class general_database_check():

    def __init__(self, output, keys, date_ref):

        # assert keys in output.columns.tolist(), 'keys must be in df'

        self.output = output
        self.keys = keys
        self.date_ref = date_ref
        self.date_min  = output[date_ref].min()
        self.date_max  = output[date_ref].max()

    def check_duplicate_keys(self):
        return 'duplicates: ' + str(self.output[self.keys].duplicated(keep=False).sum(axis=0))

    def check_stability(self):
        base = self.output[[self.date_ref]].copy()
        base['register'] = base.index
        plt.figure(figsize=(12,20))

        counts = base.groupby(self.date_ref).count()

        counts.plot(kind='line')

        plt.hlines(counts['register'].mean(),linestyles='dashed', color='black', xmin = self.date_min, xmax = self.date_max, label='mean' )
        plt.hlines(counts['register'].mean() + 2*counts['register'].std(),linestyles='dashed', color='red', xmin = self.date_min, xmax = self.date_max, label='mean +2std' )
        plt.hlines(counts['register'].mean() - 2*counts['register'].std(),linestyles='dashed', color='red', xmin = self.date_min, xmax = self.date_max, label='mean -2std' )

        plt.legend()

