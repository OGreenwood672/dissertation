import functools
import builtins



"""
The `LoggingFunctionIdentification` decorator allows you to prefix all print statements within a decorated function
with a specified identifier, making it easier to trace logs back to their origin.
"""
def LoggingFunctionIdentification(prefix):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_print = builtins.print
            def custom_print(*p_args, **p_kwargs):
                original_print(f"[{prefix}]", *p_args, **p_kwargs)
            builtins.print = custom_print
            try:
                result = func(*args, **kwargs)
            finally:
                builtins.print = original_print
            return result
        return wrapper
    return decorator
