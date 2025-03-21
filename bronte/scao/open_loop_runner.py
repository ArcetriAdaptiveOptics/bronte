from bronte.scao.flattening_runner import FlatteningRunner

class OpenLoopRunner():
    
    LOOP_TYPE = 'OPEN'
    
    def __init__(self, flattening_factory):
        
        self._factory = flattening_factory
        self._factory.INT_GAIN = 0
        self._factory.INT_DELAY = 0
        self._fr = FlatteningRunner(self._factory)
        self._fr.LOOP_TYPE = self.LOOP_TYPE
        
         
    def run(self, Nsteps = 30):
        self._fr.run(Nsteps)
    
    def save_telemetry(self, ftag):
        self._fr.save_telemetry(ftag)
        
    @staticmethod
    def load_telemetry(ftag):
        return FlatteningRunner.load_telemetry(ftag)