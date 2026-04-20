import ruptures as rpt


rup_config={'model': 'l1',
            'pen' : 1,
            'min_size' : 1, 
            'jump' : 1,
    }


class Rupture:
    def __init__(self,fret):
        
        self.fret=fret
        self.config=rup_config
        
    def det_bkps(self):
        
        config=self.config
        model = config['model']
        min_size=config['min_size']
        pen=config['pen']
        fret=self.fret
        
        print('fitting:')
        
        bkps = rpt.Binseg(model=model, min_size=min_size).fit_predict(fret,pen=pen)
        bkps = bkps[:-1]
        print(bkps)
        
        return bkps
