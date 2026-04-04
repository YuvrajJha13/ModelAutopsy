"""
Reinforced IDE Debugger - Enterprise ML Debugging Library.
Optimized for performance with multi-engine support (Rust, C++, Python fallback).
"""
import functools
import inspect
import numpy as np
from typing import Any, Dict, Optional, Callable
from rich.console import Console
from rich.table import Table

console = Console()

# 1. Load Rust Engine (Fastest, parallel processing)
try:
    import modelautopsy_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# 2. Load C++ Engine (High performance, SIMD optimized)
try:
    import _core_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

# Cache for engine priority to avoid repeated checks
_ENGINE_PRIORITY = None

def _get_engine_priority():
    """Determine engine priority once and cache the result."""
    global _ENGINE_PRIORITY
    if _ENGINE_PRIORITY is not None:
        return _ENGINE_PRIORITY
    
    if RUST_AVAILABLE:
        _ENGINE_PRIORITY = "rust"
    elif CPP_AVAILABLE:
        _ENGINE_PRIORITY = "cpp"
    else:
        _ENGINE_PRIORITY = "python"
    
    return _ENGINE_PRIORITY


def _prepare_tensor(tensor: Any) -> Optional[np.ndarray]:
    """
    Efficiently prepare tensor for analysis.
    Handles various input types with minimal copying.
    """
    if isinstance(tensor, np.ndarray):
        # In-place conversion when possible to avoid unnecessary copies
        if tensor.dtype != np.float32:
            try:
                tensor = tensor.astype(np.float32, copy=False)
            except (ValueError, TypeError):
                tensor = tensor.astype(np.float32)
        if tensor.ndim != 1:
            # Use ravel for view when possible instead of flatten (which always copies)
            tensor = tensor.ravel()
        return tensor
    
    # Handle non-numpy inputs
    try:
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.ndim != 1:
            tensor = tensor.ravel()
        return tensor
    except (ValueError, TypeError, MemoryError):
        return None


def _convert_rust_to_dict(rust_obj: Any) -> Dict[str, Any]:
    """Convert Rust StatsReport object to Python dictionary efficiently."""
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


def analyze(tensor: Any) -> Optional[Dict[str, Any]]:
    """
    Performs tensor analysis using the best available engine.
    
    Priority: Rust > C++ > Pure Python
    
    Args:
        tensor: Input array-like data (numpy array, list, etc.)
        
    Returns:
        Dictionary containing statistical analysis results, or None if input is invalid.
    """
    tensor = _prepare_tensor(tensor)
    if tensor is None:
        return None

    # Empty array handling
    if tensor.size == 0:
        return {
            "nan_count": 0,
            "inf_count": 0,
            "valid_count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "l2_norm": 0.0,
            "min_val": 0.0,
            "max_val": 0.0,
        }

    engine = _get_engine_priority()
    
    if engine == "rust":
        try:
            rust_obj = modelautopsy_rust.rust_analyze(tensor)
            return _convert_rust_to_dict(rust_obj)
        except Exception:
            # Fallback to next engine on error
            pass
    
    if engine == "cpp":
        try:
            return _core_cpp.analyze(tensor)
        except Exception:
            # Fallback to pure Python on error
            pass

    # Pure Python fallback (optimized numpy operations)
    nan_mask = np.isnan(tensor)
    inf_mask = np.isinf(tensor)
    nan_count = int(np.count_nonzero(nan_mask))
    inf_count = int(np.count_nonzero(inf_mask))
    valid_mask = ~(nan_mask | inf_mask)
    valid_count = int(np.count_nonzero(valid_mask))
    
    if valid_count > 0:
        valid_data = tensor[valid_mask]
        mean_val = float(np.mean(valid_data))
        variance_val = float(np.var(valid_data))
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        # Calculate L2 norm only on valid data to avoid NaN contamination
        l2_norm = float(np.linalg.norm(valid_data))
    else:
        mean_val = 0.0
        variance_val = 0.0
        min_val = 0.0
        max_val = 0.0
        l2_norm = 0.0
    
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "valid_count": valid_count,
        "mean": mean_val,
        "variance": variance_val,
        "l2_norm": l2_norm,
        "min_val": min_val,
        "max_val": max_val,
    }


def _log_error(func_name: str, source: str, report: Dict[str, Any]) -> None:
    """Log failure details in a formatted table."""
    console.print(f"\n[bold red]⚠️  FAILURE DETECTED[/bold red]")
    t = Table(show_header=False, header_style="bold magenta")
    t.add_row("[dim]Function:[/dim]", f"[bold]{func_name}[/bold]")
    t.add_row("[dim]Source:[/dim]", f"[bold]{source}[/bold]")
    t.add_row("[dim]NaN Count:[/dim]", f"[red]{report['nan_count']}[/red]")
    t.add_row("[dim]Inf Count:[/dim]", f"[red]{report['inf_count']}[/red]")
    t.add_row("[dim]Mean:[/dim]", f"{report['mean']:.4e}")
    t.add_row("[dim]L2 Norm:[/dim]", f"{report['l2_norm']:.4e}") 
    t.add_row("[dim]Variance:[/dim]", f"{report['variance']:.4e}") 
    console.print(t)


def watch(
    drop_into_debugger: bool = False,
    verbose: bool = True,
    inspect_args: bool = True,
    inspect_return: bool = True,
    fail_on_nan: bool = False,
    fail_on_inf: bool = False,
) -> Callable:
    """
    Enhanced decorator for monitoring function inputs and outputs.
    
    Features:
    - Automatic tensor analysis for NaN/Inf detection
    - Detailed reporting with rich formatting
    - Optional debugger breakpoint on failure
    - Configurable inspection of args and return values
    - Optional exception raising on failures
    
    Args:
        drop_into_debugger: If True, breaks into pdb on failure detection
        verbose: If True, prints status messages
        inspect_args: If True, analyzes function arguments
        inspect_return: If True, analyzes function return value
        fail_on_nan: If True, raises ValueError when NaN detected
        fail_on_inf: If True, raises ValueError when Inf detected
        
    Returns:
        Decorated function with monitoring capabilities
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if verbose:
                console.print(f"[dim]🔍 [cyan]{func.__name__}[/cyan] execution started...[/dim]")

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            failure_detected = False
            failure_details = []
            
            # Scan Inputs
            if inspect_args:
                for name, val in bound_args.arguments.items():
                    rep = analyze(val)
                    if rep:
                        has_nan = rep['nan_count'] > 0
                        has_inf = rep['inf_count'] > 0
                        
                        if has_nan or has_inf:
                            _log_error(func.__name__, f"Input '{name}'", rep)
                            failure_detected = True
                            failure_details.append((name, rep))
                            
                            if fail_on_nan and has_nan:
                                raise ValueError(f"NaN detected in input '{name}'")
                            if fail_on_inf and has_inf:
                                raise ValueError(f"Inf detected in input '{name}'")

            # Execute function
            result = func(*args, **kwargs)

            # Scan Output
            if inspect_return:
                rep = analyze(result)
                if rep:
                    has_nan = rep['nan_count'] > 0
                    has_inf = rep['inf_count'] > 0
                    
                    if has_nan or has_inf:
                        _log_error(func.__name__, "Return Value", rep)
                        failure_detected = True
                        
                        if fail_on_nan and has_nan:
                            raise ValueError(f"NaN detected in return value of '{func.__name__}'")
                        if fail_on_inf and has_inf:
                            raise ValueError(f"Inf detected in return value of '{func.__name__}'")
            
            if verbose and not failure_detected:
                console.print(f"[green]✔[/green] [dim]{func.__name__} finished clean.[/dim]")

            # Breakpoint on failure
            if failure_detected and drop_into_debugger:
                console.print("[bold red]🛑 Halting execution for debugging...[/bold red]")
                import pdb; pdb.set_trace()

            return result
        return wrapper
    return decorator


__all__ = ["watch", "analyze", "_prepare_tensor"]
