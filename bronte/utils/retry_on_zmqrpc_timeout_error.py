from plico.rpc.zmq_remote_procedure_call import ZmqRpcTimeoutError
import time

def retry_on_timeout(func, max_retries = 5000, delay = 0.005):
            '''Retries a function call if ZmqRpcTimeoutError occurs.'''
            for attempt in range(max_retries):
                try:
                    return func()
                except ZmqRpcTimeoutError:
                    print(f"Timeout error, retrying {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
            raise ZmqRpcTimeoutError("Max retries reached")