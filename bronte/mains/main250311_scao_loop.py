from bronte import startup
from bronte.scao.specula_scao_runner import SpeculaScaoRunner

def main():
    
    factory = startup.specula_startup()
    loop_time_step_in_sec = 0.001
    factory.TIME_STEP_IN_SEC = loop_time_step_in_sec
    T = 5.
    Nsteps = int(T/loop_time_step_in_sec)
    ssl = SpeculaScaoRunner(factory)
    ssl.run(Nsteps)
    ssl.save_telemetry('pippo')