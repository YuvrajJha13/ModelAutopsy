"""
Reinforced IDE Debugger.
"""
import functools
import inspect
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# 1. Load Rust Engine
try:
    import modelautopsy_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# 2. Load C++ Engine
try:
    import _core_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

def _prepare_tensor(tensor):
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = np.array(tensor)
        except:
            return None
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    if tensor.ndim != 1:
        tensor = tensor.flatten()
    return tensor

def _convert_rust_to_dict(rust_obj):
    return {
        "nan_count": rust_obj.nan_count,
        "inf_count": rust_obj.inf_count,
        "valid_count": rust_obj.valid_count,
        "mean": rust_obj.mean,
        "variance": rust_obj.variance,
        "l2_norm": rust_obj.l2_norm,
        "min_val": rust_obj.min_val,
        "max_val": rust_obj.max_val,
    }

def analyze(tensor):
    """Performs the C++ scan safely."""
    tensor = _prepare_tensor(tensor)
    if tensor is None: return None

    if RUST_AVAILABLE:
        rust_obj = modelautopsy_rust.rust_analyze(tensor)
        return _convert_rust_to_dict(rust_obj)

    if CPP_AVAILABLE:
        return _core_cpp.analyze(tensor)

    console.print("[red]Warning: Using Pure Python (Slow)[/red]")
    return {
        "nan_count": np.isnan(tensor).sum(),
        "inf_count": np.isinf(tensor).sum(),
        "valid_count": tensor.size - np.isnan(tensor).sum() - np.isinf(tensor).sum(),
        "mean": float(np.mean(tensor)) if tensor.size > 0 else 0,
        "variance": float(np.var(tensor)) if tensor.size > 0 else 0,
        "l2_norm": float(np.linalg.norm(tensor)),
        "min_val": float(np.min(tensor)) if tensor.size > 0 else 0,
        "max_val": float(np.max(tensor)) if tensor.size > 0 else 0,
    }

def _log_error(func, source, report):
    console.print(f"\n[bold red]⚠️  FAILURE DETECTED[/bold red]")
    t = Table(show_header=False, header_style="bold magenta")
    t.add_row("[dim]Function:[/dim]", f"[bold]{func}[/bold]")
    t.add_row("[dim]Source:[/dim]", f"[bold]{source}[/bold]")
    t.add_row("[dim]NaN Count:[/dim]", f"[red]{report['nan_count']}[/red]")
    t.add_row("[dim]Inf Count:[/dim]", f"[red]{report['inf_count']}[/red]")
    t.add_row("[dim]Mean:[/dim]", f"{report['mean']:.4e}")
    t.add_row("[dim]L2 Norm:[/dim]", f"{report['l2_norm']:.4e}") 
    t.add_row("[dim]Variance:[/dim]", f"{report['variance']:.4e}") 
    console.print(t)

def watch(drop_into_debugger=False, verbose=True, inspect_args=True, inspect_return=True):
    """
    Enhanced Decorator.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if verbose:
                console.print(f"[dim]🔍 [cyan]{func.__name__}[/cyan] execution started...[/dim]")

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            failure_detected = False
            
            # Scan Inputs
            if inspect_args:
                for name, val in bound_args.arguments.items():
                    rep = analyze(val)
                    if rep and (rep['nan_count'] > 0 or rep['inf_count'] > 0):
                        _log_error(func.__name__, f"Input '{name}'", rep)
                        failure_detected = True

            # Execute
            result = func(*args, **kwargs)

            # Scan Output
            if inspect_return:
                rep = analyze(result)
                if rep and (rep['nan_count'] > 0 or rep['inf_count'] > 0):
                    _log_error(func.__name__, "Return Value", rep)
                    failure_detected = True
            
            if verbose and not failure_detected:
                console.print(f"[green]✔[/green] [dim]{func.__name__} finished clean.[/dim]")

            # Breakpoint
            if failure_detected and drop_into_debugger:
                console.print("[bold red]🛑 Halting execution...[/bold red]")
                import pdb; pdb.set_trace()

            return result
        return wrapper
    return decorator
