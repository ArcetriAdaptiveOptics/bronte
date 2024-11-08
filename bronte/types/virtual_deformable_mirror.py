

class VirtualDeformableMirror():
    
    def __init__(self,
                 ID = 1,
                 name = 'pfdm',
                 Nact = 918,
                 act_pitch_in_meters = 0.02,
                 conj_altitude_in_meters = 6000 
                 ):
        self._id = ID
        self._name = name
        self._Nact = Nact
        self._hlayer = conj_altitude_in_meters
        self._act_pitch = act_pitch_in_meters #where???
        self._command = None
        self._command_offset = None